from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .config import load_config
from .inference import QwenImageEditWorker
from .logging_setup import setup_logging
from .runtime import install_runtime_hooks
from .queue_manager import InferenceQueue
from .schemas import GenerateRequest, GenerateResponse, HealthResponse, JobStatusResponse, QueueStatsResponse

logger = logging.getLogger("t2i_service.server")

config = load_config()
setup_logging(config.log_dir, config.log_level)
install_runtime_hooks()
worker = QwenImageEditWorker(config)
queue = InferenceQueue(config, worker)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logger.info(
        "Starting image-edit service | host=%s port=%d model=%s cuda=%s device_map=%s",
        config.host,
        config.port,
        config.model_path,
        config.cuda_devices,
        config.device_map,
    )
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, worker.load)
    except Exception as exc:
        logger.error("Model preload failed; service will retry on first request: %s", exc)
    await queue.start()
    yield
    await queue.stop()
    logger.info("Image-edit service stopped")


app = FastAPI(
    title="ViMax Local Image Edit Service",
    version="0.2.0",
    description=(
        "Queued local GPU image-edit service backed by Qwen-Image-Edit-2511 "
        "for ViMax adapters."
    ),
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    stats = queue.stats()
    status = "ok" if worker.is_loaded else "loading"
    if worker.load_error:
        status = "error"
    return HealthResponse(
        status=status,  # type: ignore[arg-type]
        model_loaded=worker.is_loaded,
        pipeline="QwenImageEditPlusPipeline",
        device_map=config.resolved_device_map(),
        queue_size=stats["queue_size"],
        active_job_id=stats["active_job_id"],
        cuda_devices=config.cuda_devices,
        model_path=config.model_path,
    )


@app.get("/v1/queue/stats", response_model=QueueStatsResponse)
async def queue_stats() -> QueueStatsResponse:
    stats = queue.stats()
    return QueueStatsResponse(**stats)


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str) -> JobStatusResponse:
    job = queue.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return queue.job_status(job)


@app.post("/v1/images/generations", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest) -> GenerateResponse:
    """Enqueue an image-edit job, wait for GPU slot, return edited image."""
    job = None
    try:
        job = await queue.submit(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        return await queue.wait_for_result(job, config.request_timeout_seconds)
    except RuntimeError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if job is not None:
            queue.release_job_image(job.job_id)


@app.post("/v1/images/generations/async", response_model=JobStatusResponse)
async def generate_image_async(request: GenerateRequest) -> JobStatusResponse:
    """Enqueue only; poll /v1/jobs/{job_id} for completion."""
    try:
        job = await queue.submit(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return queue.job_status(job)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Exception, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


def main() -> None:
    import uvicorn

    uvicorn.run(
        "t2i_service.server:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
