from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch
from diffusers import DiffusionPipeline
from PIL import Image

from .config import ServiceConfig, parse_size
from .schemas import GenerateRequest

logger = logging.getLogger("t2i_service.inference")


@dataclass(frozen=True)
class InferenceResult:
    image: Image.Image
    width: int
    height: int
    seed: int | None
    inference_ms: int


class QwenImageWorker:
    """Loads Qwen-Image once and runs blocking inference on a dedicated thread."""

    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self._pipe: DiffusionPipeline | None = None
        self._lock = threading.Lock()
        self._loaded = False
        self._load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> None:
        with self._lock:
            if self._loaded or self._load_error:
                return
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config.cuda_devices
            started = time.perf_counter()
            logger.info(
                "Loading model from %s on CUDA devices %s",
                self.config.model_path,
                self.config.cuda_devices,
            )
            try:
                self._pipe = DiffusionPipeline.from_pretrained(
                    self.config.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="balanced",
                )
                self._loaded = True
                elapsed = int((time.perf_counter() - started) * 1000)
                logger.info("Model loaded in %d ms", elapsed)
            except Exception as exc:
                self._load_error = str(exc)
                logger.exception("Failed to load model: %s", exc)
                raise

    def generate(self, request: GenerateRequest) -> InferenceResult:
        if not self._loaded:
            self.load()
        if self._pipe is None:
            raise RuntimeError(self._load_error or "Model is not loaded")

        width, height = self._resolve_size(request)
        negative_prompt = request.negative_prompt or self.config.default_negative_prompt
        steps = request.num_inference_steps or self.config.default_steps
        cfg_scale = request.true_cfg_scale if request.true_cfg_scale is not None else self.config.default_cfg_scale
        seed = request.seed

        if request.reference_images_b64:
            logger.warning(
                "Received %d reference image(s); Qwen-Image text-to-image path ignores them for now.",
                len(request.reference_images_b64),
            )

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda:0").manual_seed(seed)

        started = time.perf_counter()
        logger.info(
            "Inference start | size=%dx%d steps=%d cfg=%.2f seed=%s prompt=%r",
            width,
            height,
            steps,
            cfg_scale,
            seed,
            request.prompt[:120],
        )

        with self._lock:
            image = self._pipe(
                prompt=request.prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                true_cfg_scale=cfg_scale,
                generator=generator,
            ).images[0]

        inference_ms = int((time.perf_counter() - started) * 1000)
        logger.info("Inference done in %d ms", inference_ms)
        return InferenceResult(
            image=image,
            width=width,
            height=height,
            seed=seed,
            inference_ms=inference_ms,
        )

    def _resolve_size(self, request: GenerateRequest) -> tuple[int, int]:
        if request.width and request.height:
            return request.width, request.height
        return parse_size(
            request.size,
            request.aspect_ratio,
            self.config.default_width,
            self.config.default_height,
        )


def image_to_b64(image: Image.Image, image_format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format.upper())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def summarize_request(request: GenerateRequest) -> dict[str, Any]:
    width, height = request.width or 0, request.height or 0
    if not width or not height:
        width, height = parse_size(
            request.size,
            request.aspect_ratio,
            0,
            0,
        )
    return {
        "prompt_preview": request.prompt[:160],
        "width": width,
        "height": height,
        "steps": request.num_inference_steps,
        "seed": request.seed,
        "reference_images": len(request.reference_images_b64),
    }
