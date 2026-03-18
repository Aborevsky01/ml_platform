import asyncio
import time
from typing import List

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
import onnxruntime as ort
import torch
import torch.nn.functional as F
import uvicorn
from prometheus_client import make_asgi_app, Summary, Counter

MAX_BATCH_SIZE = 32
MAX_WAIT_MS = 20
NUM_WORKERS = 2  # второй воркер тут

app = FastAPI()

REQUEST_TIME = Summary("request_processing_seconds_batch", "Time spent processing request (batched ONNX)")
REQUEST_COUNT = Counter("request_count_batch", "Total number of requests (batched ONNX)")
BATCH_SIZE_HIST = Summary("batch_size_observed", "Observed batch sizes")
app.mount("/metrics", make_asgi_app())

MODEL_PATH = "rubert_mini_frida_onnx"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 2

model = ORTModelForFeatureExtraction.from_pretrained(
    MODEL_PATH,
    session_options=sess_options
)

class EmbedRequest(BaseModel):
    texts: List[str]

class RequestContext:
    def __init__(self, texts: List[str]) -> None:
        self.texts = texts
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()

request_queue: asyncio.Queue[RequestContext] = asyncio.Queue()

def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

async def batching_worker(worker_id: int) -> None:
    while True:
        first_item = await request_queue.get()
        batch: List[RequestContext] = [first_item]
        batch_texts: List[str] = list(first_item.texts)
        batch_start_time = time.perf_counter()

        while len(batch) < MAX_BATCH_SIZE:
            elapsed_ms = (time.perf_counter() - batch_start_time) * 1000
            if elapsed_ms >= MAX_WAIT_MS:
                break
            try:
                timeout = max(0.0, MAX_WAIT_MS / 1000 - (time.perf_counter() - batch_start_time))
                next_item = await asyncio.wait_for(request_queue.get(), timeout=timeout)
                batch.append(next_item)
                batch_texts.extend(next_item.texts)
            except asyncio.TimeoutError:
                break

        BATCH_SIZE_HIST.observe(len(batch_texts))

        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = mean_pooling(outputs, inputs["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings_list: List[List[float]] = embeddings.tolist()

        idx = 0
        for ctx in batch:
            n = len(ctx.texts)
            slice_embeddings = embeddings_list[idx: idx + n]
            idx += n
            if not ctx.future.done():
                ctx.future.set_result(slice_embeddings)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)

    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time

    REQUEST_TIME.observe(process_time)
    return response

@app.on_event("startup")
async def startup_event() -> None:
    loop = asyncio.get_event_loop()
    for i in range(NUM_WORKERS):
        loop.create_task(batching_worker(i))

@app.post("/embed")
async def embed(req: EmbedRequest):
    REQUEST_COUNT.inc()
    ctx = RequestContext(req.texts)
    await request_queue.put(ctx)
    embeddings = await ctx.future
    return {"embeddings": embeddings}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
