from __future__ import annotations

import os
from typing import Any, Optional

from PIL import Image


def resolve_attn(attn: str) -> Optional[str]:
    value = (attn or "auto").strip().lower()
    if value in {"none", "off", "null"}:
        return None
    if value in {"sdpa", "flash_attention_2", "eager"}:
        return value
    if value != "auto":
        raise ValueError("attn must be auto, sdpa, flash_attention_2, eager, or none")
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


class Qwen3VLDetector:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        attn: str = "auto",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 1024,
        allow_tf32: bool = False,
    ) -> None:
        self.model_id = model_id
        self.attn = attn
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.allow_tf32 = allow_tf32
        self.model = None
        self.processor = None

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if self.allow_tf32 and torch.cuda.is_available():
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        attn_impl = resolve_attn(self.attn)
        kwargs: dict[str, Any] = {
            "device_map": self._device_map_arg(),
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if attn_impl is not None:
            kwargs["attn_implementation"] = attn_impl

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_id, **kwargs)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model.eval()

    def _device_map_arg(self) -> Any:
        value = (self.device_map or "auto").strip().lower()
        if value in {"single", "cuda", "cuda:0", "0"}:
            return {"": 0}
        if value in {"auto", "balanced", "sequential"}:
            return value
        if value in {"cpu"}:
            return {"": "cpu"}
        return self.device_map

    def _input_device(self):
        import torch

        if self.model is None:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            return self.model.device
        except Exception:
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            return torch.device("cpu")

    @staticmethod
    def _move_to_device(obj: Any, device: Any) -> Any:
        import torch

        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: Qwen3VLDetector._move_to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, list):
            return [Qwen3VLDetector._move_to_device(v, device) for v in obj]
        if isinstance(obj, tuple):
            return tuple(Qwen3VLDetector._move_to_device(v, device) for v in obj)
        return obj

    def _build_inputs(self, image: Image.Image, prompt: str) -> Any:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        assert self.processor is not None
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        except Exception:
            # Fallback for processor versions that expect text + images separately.
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        if isinstance(inputs, dict):
            inputs.pop("token_type_ids", None)
        return inputs

    def generate(self, image: Image.Image, prompt: str) -> str:
        self.load()
        import torch

        assert self.model is not None
        assert self.processor is not None
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self._build_inputs(image, prompt)
        device = self._input_device()
        try:
            inputs = inputs.to(device)
        except Exception:
            inputs = self._move_to_device(inputs, device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        input_len = input_ids.shape[1]
        trimmed = generated_ids[:, input_len:]
        text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return text.strip()
