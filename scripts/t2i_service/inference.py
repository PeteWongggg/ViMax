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
from diffusers import QwenImageEditPlusPipeline
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
    reference_image_count: int
    inference_ms: int


class QwenImageEditWorker:
    """Loads Qwen-Image-Edit once and runs blocking inference on a dedicated thread."""

    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self._pipe: QwenImageEditPlusPipeline | None = None
        self._torch_device = torch.device("cpu")
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
            device_map = self.config.resolved_device_map()
            started = time.perf_counter()
            logger.info(
                "Loading QwenImageEditPlusPipeline from %s | cuda_devices=%s device_map=%s (requested=%s)",
                self.config.model_path,
                self.config.cuda_devices,
                device_map,
                self.config.device_map,
            )
            try:
                load_kwargs: dict[str, Any] = {
                    "torch_dtype": torch.bfloat16,
                }
                if device_map == "balanced":
                    load_kwargs["device_map"] = "balanced"
                self._pipe = QwenImageEditPlusPipeline.from_pretrained(
                    self.config.model_path,
                    **load_kwargs,
                )
                if device_map == "cuda":
                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA is not available after setting CUDA_VISIBLE_DEVICES")
                    self._pipe.to("cuda")
                    self._torch_device = torch.device("cuda:0")
                else:
                    self._torch_device = torch.device("cuda:0")
                self._pipe.set_progress_bar_config(disable=None)
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

        reference_images = decode_reference_images(request.reference_images_b64)
        if len(reference_images) > self.config.max_reference_images:
            raise ValueError(
                f"Too many reference images: {len(reference_images)} > {self.config.max_reference_images}"
            )

        width, height = self._resolve_size(request)
        if not reference_images:
            if self.config.require_reference_images:
                raise ValueError(
                    "At least one reference image is required for Qwen-Image-Edit. "
                    "Set T2I_REQUIRE_REFERENCE_IMAGES=false to allow blank fallback canvas."
                )
            reference_images = [blank_reference_image(width, height)]
            logger.warning(
                "No reference images provided; using blank %dx%d canvas as fallback input.",
                width,
                height,
            )

        negative_prompt = (
            request.negative_prompt
            if request.negative_prompt is not None
            else self.config.default_negative_prompt
        )
        steps = request.num_inference_steps or self.config.default_steps
        true_cfg_scale = (
            request.true_cfg_scale
            if request.true_cfg_scale is not None
            else self.config.default_cfg_scale
        )
        guidance_scale = (
            request.guidance_scale
            if request.guidance_scale is not None
            else self.config.default_guidance_scale
        )
        seed = request.seed

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._torch_device).manual_seed(seed)

        inputs: dict[str, Any] = {
            "image": reference_images,
            "prompt": request.prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "true_cfg_scale": true_cfg_scale,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": 1,
        }
        if generator is not None:
            inputs["generator"] = generator

        started = time.perf_counter()
        logger.info(
            "Inference start | refs=%d steps=%d true_cfg=%.2f guidance=%.2f seed=%s prompt=%r",
            len(reference_images),
            steps,
            true_cfg_scale,
            guidance_scale,
            seed,
            request.prompt[:120],
        )

        with self._lock:
            with torch.inference_mode():
                output = self._pipe(**inputs)
            image = output.images[0]

        inference_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "Inference done in %d ms | output=%dx%d",
            inference_ms,
            image.width,
            image.height,
        )
        return InferenceResult(
            image=image,
            width=image.width,
            height=image.height,
            seed=seed,
            reference_image_count=len(reference_images),
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


def decode_reference_images(reference_images_b64: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for index, encoded in enumerate(reference_images_b64):
        if not encoded:
            continue
        try:
            raw = base64.b64decode(encoded, validate=True)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Invalid reference image at index {index}: {exc}") from exc
        images.append(image)
    return images


def blank_reference_image(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), color=(255, 255, 255))


def image_to_b64(image: Image.Image, image_format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format.upper())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def summarize_request(request: GenerateRequest) -> dict[str, Any]:
    width, height = request.width or 0, request.height or 0
    if not width or not height:
        width, height = parse_size(request.size, request.aspect_ratio, 0, 0)
    return {
        "prompt_preview": request.prompt[:160],
        "width_hint": width or None,
        "height_hint": height or None,
        "steps": request.num_inference_steps,
        "true_cfg_scale": request.true_cfg_scale,
        "guidance_scale": request.guidance_scale,
        "seed": request.seed,
        "reference_images": len(request.reference_images_b64),
    }
