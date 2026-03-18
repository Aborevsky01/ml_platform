from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
import onnxruntime as ort
import torch
import torch.nn.functional as F
import uvicorn
import time
from prometheus_client import make_asgi_app, Summary, Counter

app = FastAPI()

REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNT = Counter('request_count', 'Total number of requests')
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

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)
        
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    
    REQUEST_TIME.observe(process_time)
    print(f"ONNX Request processed in {process_time * 1000:.2f} ms")
    return response

class EmbedRequest(BaseModel):
    texts: list[str]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


@app.post("/embed")
def embed(req: EmbedRequest):
    REQUEST_COUNT.inc()
    inputs = tokenizer(req.texts, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    embeddings = mean_pooling(outputs, inputs['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return {"embeddings": embeddings.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)