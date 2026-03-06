from itertools import chain
import json
import logging
import os
from typing import Dict, List, Any
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated
import select
import uuid

logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/Step-Audio2/main.py")
class StepAudio2(OfflineModel):
    def __init__(
        self,
        model_path: str = "stepfun-ai/Step-Audio-2-mini",
        prompt_wav: str = "assets/default_female.wav",
        speech: bool = False,
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        if not os.path.exists(model_path):
             logger.warning(f"Model path {model_path} does not exist locally.")

        self.command_args = {
            "model_path": model_path,
            "prompt_wav": prompt_wav,
        }
        if speech:
            self.command_args["speech"] = ""

        super().__init__(is_chat=True, sample_params=sample_params)

    def _format_messages(self, prompt: PromptStruct) -> List[Dict[str, Any]]:
        """
        将 PromptStruct 转换为 StepAudio2 接受的 standard message list 格式。
        StepAudio2 接受:
        1. 文本: {"role": "human", "content": "text"}
        2. 音频: {"role": "human", "content": [{"type": "audio", "audio": "path"}]}
        """
        messages = []
        
        for turn in prompt:
            # audio_evals 通常使用 "user", StepAudio 偏好 "human" (API 脚本中已做兼容，但这里转换更保险)
            role = turn["role"]
            if role == "user":
                role = "human"
                
            contents = turn["contents"]
            
            # 检查是否为纯文本内容
            is_text_only = all(c["type"] == "text" for c in contents)
            
            if is_text_only:
                # 拼接多段文本
                text_value = "".join([c["value"] for c in contents])
                messages.append({"role": role, "content": text_value})
            else:
                # 包含音频或其他模态
                # StepAudio 的音频输入通常是 list of dicts
                mixed_content = []
                for c in contents:
                    if c["type"] == "audio":
                        mixed_content.append({"type": "audio", "audio": c["value"]})
                    elif c["type"] == "text":
                        mixed_content.append({"type": "text", "text": c["value"]})
                
                if mixed_content:
                    messages.append({"role": role, "content": mixed_content})

        return messages

    def _inference(self, prompt: PromptStruct, **kwargs):
        # 1. 格式化输入消息
        valid_messages = self._format_messages(prompt)

        # 2. 准备通信协议头
        uid = str(uuid.uuid4())
        prefix = f"{uid}->"
        payload = {"messages": valid_messages}
        
        # 3. 发送请求 (Stdin)
        while True:
            # 检查 stdin 是否可写 (超时 180s)
            _, wlist, _ = select.select([], [self.process.stdin], [], 180)
            if wlist:
                try:
                    # 写入 JSON 数据
                    self.process.stdin.write(f"{prefix}{json.dumps(payload)}\n")
                    self.process.stdin.flush()
                    break
                except BrokenPipeError:
                    logger.error("StepAudio process pipe broken.")
                    raise RuntimeError("StepAudio process crashed.")
            else:
                raise RuntimeError("Timeout waiting to write to StepAudio process.")

        # 4. 读取响应 (Stdout)
        while True:
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 180
            )
            if not rlist:
                err_msg = "Read timeout after 180 seconds waiting for StepAudio response"
                logger.error(err_msg)
                raise RuntimeError(err_msg)
            
            try:
                for stream in rlist:
                    if stream == self.process.stdout:
                        result = self.process.stdout.readline().strip()
                        if not result:
                            continue
                        if result.startswith(prefix):
                            self.process.stdin.write(f"{prefix}close\n")
                            self.process.stdin.flush()
                            res = json.loads(result[len(prefix) :])
                            if len(res) == 1:
                                return res["text"]
                            return result[len(prefix) :]
                            
                        elif result.startswith("Error:"):
                            raise RuntimeError(f"StepAudio API Error: {result}")
                        else:
                            logger.info(f"[StepAudio Worker]: {result}")
                    elif stream == self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            logger.error(f"[StepAudio Stderr]: {err}")
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {str(e)}")