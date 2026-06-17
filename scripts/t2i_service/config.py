from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    return float(raw)


DEFAULT_NEGATIVE_PROMPT = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，"
    "过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
)

ASPECT_RATIO_SIZES: dict[str, tuple[int, int]] = {
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "1:1": (1328, 1328),
}


@dataclass(frozen=True)
class ServiceConfig:
    model_path: str = field(
        default_factory=lambda: _env(
            "T2I_MODEL_PATH",
            "/nas/models/t2i/qwen-image-2512/Qwen/Qwen-Image-2512",
        )
    )
    cuda_devices: str = field(default_factory=lambda: _env("T2I_CUDA_DEVICES", "0,1"))
    host: str = field(default_factory=lambda: _env("T2I_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("T2I_PORT", 8100))
    max_queue_size: int = field(default_factory=lambda: _env_int("T2I_MAX_QUEUE_SIZE", 32))
    request_timeout_seconds: float = field(
        default_factory=lambda: _env_float("T2I_REQUEST_TIMEOUT_SECONDS", 600.0)
    )
    default_negative_prompt: str = field(
        default_factory=lambda: _env("T2I_DEFAULT_NEGATIVE_PROMPT", DEFAULT_NEGATIVE_PROMPT)
    )
    default_width: int = field(default_factory=lambda: _env_int("T2I_DEFAULT_WIDTH", 1664))
    default_height: int = field(default_factory=lambda: _env_int("T2I_DEFAULT_HEIGHT", 928))
    default_steps: int = field(default_factory=lambda: _env_int("T2I_DEFAULT_STEPS", 50))
    default_cfg_scale: float = field(default_factory=lambda: _env_float("T2I_DEFAULT_CFG_SCALE", 4.0))
    log_dir: Path = field(
        default_factory=lambda: Path(_env("T2I_LOG_DIR", "logs/t2i_service")).resolve()
    )
    log_level: str = field(default_factory=lambda: _env("T2I_LOG_LEVEL", "INFO").upper())
    output_format: str = field(default_factory=lambda: _env("T2I_OUTPUT_FORMAT", "png").lower())

    def validate(self) -> None:
        if not self.model_path:
            raise ValueError("T2I_MODEL_PATH is required")
        if self.max_queue_size < 1:
            raise ValueError("T2I_MAX_QUEUE_SIZE must be >= 1")
        if self.port < 1 or self.port > 65535:
            raise ValueError("T2I_PORT must be between 1 and 65535")


def parse_size(size: str | None, aspect_ratio: str | None, default_width: int, default_height: int) -> tuple[int, int]:
    if size:
        normalized = size.lower().replace(" ", "")
        if "x" in normalized:
            width_text, height_text = normalized.split("x", 1)
            return int(width_text), int(height_text)
    if aspect_ratio and aspect_ratio in ASPECT_RATIO_SIZES:
        return ASPECT_RATIO_SIZES[aspect_ratio]
    return default_width, default_height


def load_config() -> ServiceConfig:
    config = ServiceConfig()
    config.validate()
    return config
