from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
import torch
import time
from prometheus_client import make_asgi_app, Summary, Counter

app = FastAPI()

REQUEST_TIME = Summary('request_processing_seconds_pt', 'Time spent processing request (PyTorch)')
REQUEST_COUNT = Counter('request_count_pt', 'Total number of requests (PyTorch)')
app.mount("/metrics", make_asgi_app())

model_name = "sergeyzh/rubert-mini-frida"
model = SentenceTransformer(model_name, device="cpu")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)

    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time

    REQUEST_TIME.observe(process_time)
    print(f"PT Request processed in {process_time * 1000:.2f} ms")
    return response

class EmbedRequest(BaseModel):
    texts: list[str]

@app.post("/embed")
def embed(req: EmbedRequest):
    REQUEST_COUNT.inc()
    with torch.no_grad():
        embeddings = model.encode(req.texts, normalize_embeddings=True)
    return {"embeddings": embeddings.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
