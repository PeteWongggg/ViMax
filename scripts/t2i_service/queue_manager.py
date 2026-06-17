from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .config import ServiceConfig
from .inference import QwenImageEditWorker, image_to_b64, summarize_request
from .schemas import GenerateRequest, GenerateResponse, JobStatusResponse

logger = logging.getLogger("t2i_service.queue")
request_logger = logging.getLogger("t2i_service.requests")


@dataclass
class QueueJob:
    job_id: str
    request: GenerateRequest
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    status: str = "queued"
    error: str | None = None
    result: GenerateResponse | None = None
    queue_wait_ms: int = 0
    future: asyncio.Future[GenerateResponse] = field(default_factory=asyncio.Future)


class InferenceQueue:
    def __init__(self, config: ServiceConfig, worker: QwenImageWorker) -> None:
        self.config = config
        self.worker = worker
        self._queue: asyncio.Queue[QueueJob | None] | None = None
        self._jobs: dict[str, QueueJob] = {}
        self._active_job_id: str | None = None
        self._completed_jobs = 0
        self._failed_jobs = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="t2i-infer")
        self._consumer_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._consumer_task is not None:
            return
        self._queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self._consumer_task = asyncio.create_task(self._consume(), name="t2i-queue-consumer")
        logger.info("Queue started | max_size=%d", self.config.max_queue_size)

    async def stop(self) -> None:
        if self._queue is not None:
            await self._queue.put(None)
        if self._consumer_task is not None:
            await self._consumer_task
        self._executor.shutdown(wait=True, cancel_futures=False)
        logger.info("Queue stopped")

    async def submit(self, request: GenerateRequest) -> QueueJob:
        if self._queue is None:
            raise RuntimeError("Queue is not started")
        job_id = str(uuid.uuid4())
        job = QueueJob(job_id=job_id, request=request)
        self._jobs[job_id] = job
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            self._jobs.pop(job_id, None)
            raise RuntimeError(f"Queue is full (max={self.config.max_queue_size})") from exc
        request_logger.info("queued | job_id=%s | %s", job_id, summarize_request(request))
        return job

    async def wait_for_result(self, job: QueueJob, timeout_seconds: float) -> GenerateResponse:
        try:
            return await asyncio.wait_for(job.future, timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"Request timed out after {timeout_seconds:.0f}s") from exc

    def get_job(self, job_id: str) -> QueueJob | None:
        return self._jobs.get(job_id)

    def stats(self) -> dict[str, Any]:
        queue_size = self._queue.qsize() if self._queue is not None else 0
        return {
            "queue_size": queue_size,
            "max_queue_size": self.config.max_queue_size,
            "active_job_id": self._active_job_id,
            "completed_jobs": self._completed_jobs,
            "failed_jobs": self._failed_jobs,
        }

    def job_status(self, job: QueueJob) -> JobStatusResponse:
        queue_position = None
        if job.status == "queued" and self._queue is not None:
            queue_position = self._queue.qsize()
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status,  # type: ignore[arg-type]
            queue_position=queue_position,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            error=job.error,
            result=job.result,
        )

    async def _consume(self) -> None:
        assert self._queue is not None
        while True:
            job = await self._queue.get()
            if job is None:
                self._queue.task_done()
                break
            await self._run_job(job)
            self._queue.task_done()

    async def _run_job(self, job: QueueJob) -> None:
        self._active_job_id = job.job_id
        job.status = "running"
        job.started_at = time.time()
        job.queue_wait_ms = int((job.started_at - job.created_at) * 1000)
        request_logger.info(
            "running | job_id=%s | queue_wait_ms=%d",
            job.job_id,
            job.queue_wait_ms,
        )
        loop = asyncio.get_running_loop()
        total_started = time.perf_counter()
        try:
            inference_result = await loop.run_in_executor(
                self._executor,
                self.worker.generate,
                job.request,
            )
            image_b64 = image_to_b64(inference_result.image, self.config.output_format)
            total_ms = int((time.perf_counter() - total_started) * 1000)
            response = GenerateResponse(
                job_id=job.job_id,
                status="completed",
                image_b64=image_b64,
                width=inference_result.width,
                height=inference_result.height,
                format=self.config.output_format,
                seed=inference_result.seed,
                reference_image_count=inference_result.reference_image_count,
                queue_wait_ms=job.queue_wait_ms,
                inference_ms=inference_result.inference_ms,
                total_ms=total_ms,
                metadata=job.request.metadata,
            )
            job.result = response
            job.status = "completed"
            job.finished_at = time.time()
            self._completed_jobs += 1
            request_logger.info(
                "completed | job_id=%s | queue_wait_ms=%d | inference_ms=%d | total_ms=%d",
                job.job_id,
                job.queue_wait_ms,
                inference_result.inference_ms,
                total_ms,
            )
            if not job.future.done():
                job.future.set_result(response)
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = time.time()
            self._failed_jobs += 1
            request_logger.exception("failed | job_id=%s | error=%s", job.job_id, exc)
            if not job.future.done():
                job.future.set_exception(exc)
        finally:
            self._active_job_id = None
