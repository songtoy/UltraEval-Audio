import argparse
import json
import select
import sys
import tempfile
import logging
import time
import os
import torch
import traceback

from stepaudio2 import StepAudio2
from token2wav import Token2wav

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default="stepfun-ai/Step-Audio-2-mini",
        help="Path or HF repo for Step-Audio-2 model",
    )
    parser.add_argument(
        "--prompt_wav",
        type=str,
        default="assets/default_female.wav", 
        help="Reference audio file for voice cloning (speaker prompt)",
    )
    parser.add_argument(
        "--speech",
        action="store_true",
        default=False,
        help="Whether to generate speech output",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help="Whether to thinking process",
    )
    parser.add_argument(
        "--replace_think_token",
        action="store_true",
        default=False,
        help="Whether to replace <think> token with actual thinking content in the final response",
    )
    parser.add_argument(
        "--system_think_prompt",
        action="store_true",
        default=False,
        help="Whether to add extra instructions in system prompt to guide the model to generate more detailed thinking process",
    )
    args = parser.parse_args()

    start_time = time.time()
    
    model = StepAudio2(args.model_path)
    token2wav = Token2wav(os.path.join(args.model_path, "token2wav"))

    if not os.path.exists(args.prompt_wav):
        CUR_DIR = os.path.dirname(os.path.abspath(__file__))
        prompt_wav = os.path.join(CUR_DIR, "assets/default_female.wav")
        
    end_time = time.time()
    logger.info(f"Model loading took {end_time - start_time:.2f} seconds")
    logger.info(f"Using Step-Audio-2 model: {args.model_path}")

    # StepAudio2 的采样参数
    sampling_params = {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
    }
    system_think_prompt = "你的名字叫小跃，你是由阶跃星辰(StepFun)公司训练出来的语音大模型，你能听见用户的声音特征并在思维过程中描述出来，请激活深度思考模式，通过逐步分析、逻辑推理来解决用户的问题。"

    while True:
        try:
            prompt = input()

            print("prompt:", prompt)
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    "Error: Invalid conversation format, must contains  ->, but {}".format(
                        prompt
                    ),
                    flush=True,
                )
                continue

            # 2. 解析前缀和 JSON
            prefix = prompt[:anchor].strip() + "->"
            x = json.loads(prompt[anchor + 2 :])

            # 3. 准备消息历史
            messages = x["messages"] if "messages" in x else x

            print("Parsed messages: ", messages, flush=True)

            if args.system_think_prompt:
                messages.insert(0, {"role": "system", "content": system_think_prompt})

            think_start = "<think>"
            think_end = "</think>"
            if args.replace_think_token:
                think_start = "<|THINK_START|>"
                think_end = "<|THINK_END|>"

            if args.thinking:
                # 在消息历史中插入思考过程占位符
                messages.append({"role": "assistant", "content": think_start, "eot": False})
                _, think_content, _ = model(messages, stop_strings=[think_end], **sampling_params)
                print(think_start + think_content + think_end)
                # 将思考内容插入消息历史
                if args.speech:
                    think_content = think_content.replace('</think', '')
                    messages[-1]["content"] += think_content + think_end + "\n\n<tts_start>"
            else:
                # 4. 设置语音生成触发器
                if args.speech:
                    messages.append({"role": "assistant", "content": "<tts_start>","eot": False})
                else:
                    messages.append({"role": "assistant", "content": None})

            #print("messages: ", messages, flush=True)
            # 5. 推理
            # StepAudio2.__call__ 返回 (tokens, text, audio_tokens)
            tokens, text, audio_tokens = model(
                messages, 
                **sampling_params
            )

            print(">> Output: ", text)

            response_payload = {"text": text}
            if args.thinking:
                response_payload["think_content"] = think_content 

            if args.speech and audio_tokens is not None:
                try:
                    # 解码音频 Tokens 为波形 bytes
                    audio = [x for x in audio if x < 6561] # remove audio padding

                    audio_wav_bytes = token2wav(audio_tokens, prompt_wav)
                    
                    # 写入临时文件
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        f.write(audio_wav_bytes)
                        f.flush() # 确保写入磁盘
                        response_payload["audio"] = f.name
                except Exception as e:
                    logger.error(f"Audio decoding failed: {e}")
                    # 如果音频失败，只返回文本，不中断流程

            # 7. 发送响应并等待握手 (Close Signal)
            retry = 3
            while retry:
                print(
                    f"{prefix}{json.dumps(response_payload, ensure_ascii=False)}",
                    flush=True,
                )
                
                # 等待 stdin 的确认信号
                rlist, _, _ = select.select([sys.stdin], [], [], 1) # 增加超时时间防止过快
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == f"{prefix}close":
                        break
                logger.info("not found close signal, will emit again")
                retry -= 1

        except Exception as e:
            traceback.print_exc()
            print(f"Error: {str(e)}", flush=True)