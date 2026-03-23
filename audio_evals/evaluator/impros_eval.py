import ast
import os.path
import re
from copy import deepcopy
from typing import Dict
import yaml
import json
from audio_evals.evaluator.base import Evaluator


class ImprosBenchS2TEvaluator(Evaluator):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        from audio_evals.registry import registry

        model = registry.get_model(self.model_name)
        prompt = registry.get_prompt("impros-chatbot-eval")

        task = kwargs.get("task", None)
        p = prompt.load(
            user_transcription=task["speech"],
            user_emotion=task["analysis"],
            user_sarcasm="none",
            user_age=kwargs.get("age", "unknown"),
            user_gender=kwargs.get("gender", "unknown"),
            response=pred
        )
        res = model.inference(p, temperature=0, max_tokens=2048)

        # res_d = re.search(r"```json(.*?)```", res, re.DOTALL)
        d = re.search(r'\[\[(\d+)\]\]', res)
        return {
            "geval": int(d.group(1)),
            "pred": pred,
            "ref": label,
        }


class ImprosBenchS2SEvaluator(Evaluator):
    """直接使用 Speech Language Model 评估音频响应"""
    
    def __init__(self, model_name: str = "gemini-2.5-pro"):
        self.model_name = model_name

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        from audio_evals.registry import registry

        # pred 应该是音频文件路径
        model = registry.get_model(self.model_name)
        prompt = registry.get_prompt("impros-chatbot-eval-audio")

        task = kwargs.get("task", None)
        p = prompt.load(
            user_transcription=task["speech"],
            user_emotion=task["analysis"],
            user_sarcasm="none",
            user_age=kwargs.get("age", "unknown"),
            user_gender=kwargs.get("gender", "unknown"),
            response_audio=pred  # 直接传入音频路径
        )

        res = model.inference(p, temperature=0, max_tokens=2048)

        d = re.search(r'\[\[(\d+)\]\]', res)
        return {
            "geval": int(d.group(1)),
            "pred": pred,
            "ref": label,
            "pred_caption": res,
        }

class ImprosHardS2SEvaluator(Evaluator):
    """直接使用 Speech Language Model 评估音频响应"""
    
    def __init__(self, model_name: str = "gemini-2.5-pro"):
        self.model_name = model_name

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        from audio_evals.registry import registry

        # pred 应该是音频文件路径
        model = registry.get_model(self.model_name)
        prompt = registry.get_prompt("impros-chatbot-eval-audio")

        task = kwargs#.get("task", None)
        p = prompt.load(
            user_transcription=task["content"],
            user_emotion=task["tone"],
            user_sarcasm="{}. Implicate {}".format(kwargs["category"], kwargs["implication"]),
            user_age=kwargs.get("age", "unknown"),
            user_gender=kwargs.get("gender", "unknown"),
            response_audio=pred  # 直接传入音频路径
        )

        res = model.inference(p, temperature=0, max_tokens=2048)

        d = re.search(r'\[\[(\d+)\]\]', res)
        return {
            "geval": int(d.group(1)),
            "pred": pred,
            "ref": label,
            "pred_caption": res,
        }
