# ViMax Local Image Edit Service API

基于 **Qwen-Image-Edit-2511**（`QwenImageEditPlusPipeline`）的本地 GPU 图像编辑服务 HTTP API 说明。

- **默认地址**：`http://118.196.65.175:8911`
- **Content-Type**：`application/json`
- **交互文档**：`http://<host>:<port>/docs`（FastAPI Swagger）

启动服务：

```bash
./scripts/start_t2i_service.sh
```

---

## 端点一览

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `GET` | `/v1/queue/stats` | 队列统计 |
| `POST` | `/v1/images/generations` | **同步生图**（入队 → 等待完成 → 返回图片） |
| `POST` | `/v1/images/generations/async` | **异步入队**（立即返回 `job_id`） |
| `GET` | `/v1/jobs/{job_id}` | 查询异步任务状态 |

---

## 核心接口：`POST /v1/images/generations`

同步调用。HTTP 连接会保持到推理完成（默认最长等待 600 秒，由 `T2I_REQUEST_TIMEOUT_SECONDS` 控制）。

### Request Body

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `prompt` | `string` | ✅ | — | 编辑/生成提示词，不能为空 |
| `reference_images_b64` | `string[]` | ❌ | `[]` | 参考图 base64 列表（PNG/JPEG 原始字节编码），对应 pipeline 的 `image=[...]` |
| `negative_prompt` | `string` | ❌ | `" "` | 负向提示词 |
| `num_inference_steps` | `int` | ❌ | `40` | 推理步数，范围 1–150 |
| `true_cfg_scale` | `float` | ❌ | `4.0` | CFG 强度，范围 0.0–20.0 |
| `guidance_scale` | `float` | ❌ | `1.0` | Guidance scale（该模型非 guidance-distilled 时会被忽略） |
| `seed` | `int` | ❌ | 随机 | 随机种子，≥ 0 |
| `width` | `int` | ❌ | — | 仅**无参考图**时：空白画布宽度（256–4096） |
| `height` | `int` | ❌ | — | 仅**无参考图**时：空白画布高度（256–4096） |
| `size` | `string` | ❌ | — | ViMax 风格尺寸，如 `"1600x900"`，仅无参考图时用于空白画布 |
| `aspect_ratio` | `string` | ❌ | — | 宽高比 hint，仅无参考图时生效，见下表 |
| `response_format` | `string` | ❌ | `"b64_json"` | 目前仅支持 `"b64_json"`，`"url"` 未实现 |
| `metadata` | `object` | ❌ | `{}` | 透传元数据，原样回传到 response |

#### `aspect_ratio` 预设（仅无参考图时）

| 值 | 分辨率 |
|----|--------|
| `16:9` | 1664 × 928 |
| `9:16` | 928 × 1664 |
| `4:3` | 1472 × 1104 |
| `3:4` | 1104 × 1472 |
| `1:1` | 1328 × 1328 |

未指定尺寸时，默认空白画布为 **1664 × 928**。

#### 参考图行为

- **有参考图**：按列表顺序传入 pipeline；输出尺寸由模型根据参考图决定（`width/height/size` 不传给 pipeline）
- **无参考图**：
  - 默认：自动生成白色空白画布作为输入
  - 若 `T2I_REQUIRE_REFERENCE_IMAGES=true`：请求失败
- **上限**：最多 8 张（`T2I_MAX_REFERENCE_IMAGES`）

#### 尺寸 hint 解析优先级（仅无参考图）

1. `width` + `height`
2. `size`（如 `"1600x900"`）
3. `aspect_ratio`
4. 服务默认 1664×928

---

### Request 样例

#### 单张参考图编辑

```json
{
  "prompt": "基于当前人物肖像，生成一张健身照.",
  "negative_prompt": "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。",
  "reference_images_b64": [
    "<BASE64_ENCODED_PNG_OR_JPEG>"
  ],
  "num_inference_steps": 40,
  "true_cfg_scale": 4.0,
  "guidance_scale": 1.0,
  "seed": 42
}
```

#### 多张参考图编辑

```json
{
  "prompt": "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square.",
  "negative_prompt": "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。",
  "reference_images_b64": [
    "<BASE64_IMAGE_1>",
    "<BASE64_IMAGE_2>"
  ],
  "num_inference_steps": 40,
  "true_cfg_scale": 4.0,
  "guidance_scale": 1.0,
  "seed": 0
}
```

#### 无参考图（空白画布 fallback）

```json
{
  "prompt": "一个成年人，健身照，写实风格",
  "size": "1664x928",
  "seed": 42
}
```

#### ViMax 分镜帧风格

```json
{
  "prompt": "Image 0: front view of Alice\nImage 1: gym background\nGenerate a medium shot of Alice exercising.",
  "reference_images_b64": [
    "<BASE64_CHAR_PORTRAIT>",
    "<BASE64_PREVIOUS_FRAME>"
  ],
  "size": "1600x900",
  "num_inference_steps": 40,
  "true_cfg_scale": 4.0,
  "seed": 42,
  "metadata": {
    "source": "vimax",
    "shot_idx": 3
  }
}
```

#### curl

```bash
curl -sS --max-time 600 \
  -H "Content-Type: application/json" \
  -X POST "http://127.0.0.1:8911/v1/images/generations" \
  -d '{
    "prompt": "基于当前人物肖像，生成一张健身照.",
    "negative_prompt": "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。",
    "reference_images_b64": ["<BASE64>"],
    "num_inference_steps": 40,
    "true_cfg_scale": 4.0,
    "guidance_scale": 1.0,
    "seed": 42
  }'
```

---

### Response（成功，HTTP 200）

```json
{
  "job_id": "9b577b60-bb8e-41a9-950b-2d8f2b870f1e",
  "status": "completed",
  "image_b64": "<BASE64_PNG>",
  "width": 1024,
  "height": 1024,
  "format": "png",
  "seed": 42,
  "reference_image_count": 1,
  "queue_wait_ms": 5,
  "inference_ms": 18234,
  "total_ms": 18250,
  "metadata": {
    "source": "vimax",
    "shot_idx": 3
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `job_id` | `string` | 任务 UUID |
| `status` | `"completed"` | 同步接口成功时固定为 `completed` |
| `image_b64` | `string` | PNG 图片的 base64 编码 |
| `width` | `int` | 实际输出图片宽度 |
| `height` | `int` | 实际输出图片高度 |
| `format` | `string` | 图片格式，默认 `"png"` |
| `seed` | `int \| null` | 使用的随机种子 |
| `reference_image_count` | `int` | 实际送入模型的参考图数量 |
| `queue_wait_ms` | `int` | 排队等待时间（毫秒） |
| `inference_ms` | `int` | 纯 GPU 推理时间（毫秒） |
| `total_ms` | `int` | worker 开始到结束的总时间（毫秒） |
| `metadata` | `object` | 回传请求中的 `metadata` |

解码示例（Python）：

```python
import base64
from pathlib import Path

image_bytes = base64.b64decode(response["image_b64"])
Path("output.png").write_bytes(image_bytes)
```

---

### 错误 Response

```json
{
  "detail": "错误描述信息"
}
```

| HTTP 状态码 | 场景 |
|-------------|------|
| `400` | 请求体校验失败（如 `prompt` 为空） |
| `404` | `GET /v1/jobs/{job_id}` 任务不存在 |
| `500` | 推理失败（显存不足、模型错误等） |
| `503` | 队列已满 |
| `504` | 同步请求超时 |

失败时**没有** `image_b64` 字段。

---

## 异步接口

### `POST /v1/images/generations/async`

Request body 与同步接口相同。立即返回：

```json
{
  "job_id": "9b577b60-bb8e-41a9-950b-2d8f2b870f1e",
  "status": "queued",
  "queue_position": 1,
  "created_at": 1718617480.123,
  "started_at": null,
  "finished_at": null,
  "error": null,
  "result": null
}
```

### `GET /v1/jobs/{job_id}`

轮询直到 `status` 为 `completed` 或 `failed`。

完成时 `result` 字段结构与同步 Response 相同。`status` 流转：`queued` → `running` → `completed` | `failed`。

---

## 辅助接口

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "pipeline": "QwenImageEditPlusPipeline",
  "device_map": "cuda",
  "queue_size": 0,
  "active_job_id": null,
  "cuda_devices": "1",
  "model_path": "/nas/models/i2i/qwen/Qwen/Qwen-Image-Edit-2511"
}
```

| `status` | 含义 |
|----------|------|
| `ok` | 模型已加载 |
| `loading` | 服务已启动，模型尚在加载 |
| `error` | 模型加载失败 |

### `GET /v1/queue/stats`

```json
{
  "queue_size": 2,
  "max_queue_size": 32,
  "active_job_id": "abc-123",
  "completed_jobs": 15,
  "failed_jobs": 1
}
```

---

## ViMax 接入映射

| ViMax `generate_single_image()` | API 字段 |
|-----------------------------------|---------|
| `prompt` | `prompt` |
| `reference_image_paths` | `reference_images_b64`（适配器读文件 → base64） |
| `size` | `size` |
| `aspect_ratio` | `aspect_ratio` |
| 返回 `ImageOutput(fmt="pil")` | 从 `image_b64` decode |

---

## 服务行为

1. **队列串行**：GPU 同时只跑 1 个推理任务
2. **模型常驻显存**：启动时加载，进程退出才释放
3. **单卡**：`T2I_CUDA_DEVICES="1"` 时自动使用 `cuda` 模式（`pipeline.to("cuda")`）
4. **双卡**：`T2I_CUDA_DEVICES="0,1"` + `T2I_DEVICE_MAP=balanced`
5. **推理耗时**：通常 15–60+ 秒/张，客户端需设置足够长的 HTTP timeout
6. **推荐参数**：`steps=40`, `true_cfg_scale=4.0`, `guidance_scale=1.0`, `negative_prompt="低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"`

---

## 环境变量

详见 `scripts/t2i_service.env.example`。常用项：

| 变量 | 默认 | 说明 |
|------|------|------|
| `T2I_MODEL_PATH` | `/nas/models/i2i/qwen/Qwen/Qwen-Image-Edit-2511` | 模型路径 |
| `T2I_CUDA_DEVICES` | `0,1` | 可见 GPU |
| `T2I_DEVICE_MAP` | `balanced` | 单卡时自动 fallback 为 `cuda` |
| `T2I_PORT` | `8100` | 监听端口 |
| `T2I_REQUEST_TIMEOUT_SECONDS` | `600` | 同步接口超时 |
| `T2I_MAX_REFERENCE_IMAGES` | `8` | 最大参考图数 |
| `T2I_REQUIRE_REFERENCE_IMAGES` | `false` | 是否强制要求参考图 |

---

## 最小调用流程

**同步：**

```
GET  /health
POST /v1/images/generations
解析 image_b64 → 保存图片
```

**异步：**

```
POST /v1/images/generations/async
GET  /v1/jobs/{job_id}  （轮询）
解析 result.image_b64
```

本地测试脚本：`scripts/test_t2i_service_curl.sh`
