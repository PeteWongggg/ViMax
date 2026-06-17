from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Edit/generation prompt.")
    negative_prompt: str | None = Field(default=None, description="Negative prompt.")
    width: int | None = Field(
        default=None,
        ge=256,
        le=4096,
        description="Hint for blank fallback canvas when no reference images are provided.",
    )
    height: int | None = Field(
        default=None,
        ge=256,
        le=4096,
        description="Hint for blank fallback canvas when no reference images are provided.",
    )
    size: str | None = Field(default=None, description='ViMax-style size hint, e.g. "1600x900".')
    aspect_ratio: str | None = Field(default=None, description='Aspect ratio hint, e.g. "16:9".')
    num_inference_steps: int | None = Field(default=None, ge=1, le=150)
    true_cfg_scale: float | None = Field(default=None, ge=0.0, le=20.0)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=20.0)
    seed: int | None = Field(default=None, ge=0)
    reference_images_b64: list[str] = Field(
        default_factory=list,
        description="Reference images as base64 PNG/JPEG. Passed to Qwen-Image-Edit as `image=[...]`.",
    )
    response_format: Literal["b64_json", "url"] = "b64_json"
    metadata: dict[str, Any] = Field(default_factory=dict, description="Opaque metadata echoed in response.")

    @field_validator("prompt")
    @classmethod
    def strip_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt cannot be empty")
        return value


class GenerateResponse(BaseModel):
    job_id: str
    status: Literal["completed"]
    image_b64: str
    width: int
    height: int
    format: str
    seed: int | None
    reference_image_count: int
    queue_wait_ms: int
    inference_ms: int
    total_ms: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    queue_position: int | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: GenerateResponse | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    model_loaded: bool
    pipeline: str
    device_map: str
    queue_size: int
    active_job_id: str | None
    cuda_devices: str
    model_path: str


class QueueStatsResponse(BaseModel):
    queue_size: int
    max_queue_size: int
    active_job_id: str | None
    completed_jobs: int
    failed_jobs: int
