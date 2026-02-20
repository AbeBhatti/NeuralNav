"""
Conductor AWS — FastAPI endpoint
POST /chat  → routes to best model, returns response + metadata
POST /break/{model_name}   → simulates model degradation for demo
POST /restore/{model_name} → restores model and re-benchmarks
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn

from router_agent import benchmark_all_models, chat, models, consecutive_drops


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[CONDUCTOR-AWS] Starting up — running initial benchmark...")
    benchmark_all_models()
    print("[CONDUCTOR-AWS] Ready to route requests.")
    yield

app = FastAPI(title="Conductor AWS", lifespan=lifespan)


class ChatRequest(BaseModel):
    prompt:        str   = Field(..., description="The user prompt to route")
    latency_w:     float = Field(default=0.4, ge=0, description="Weight for latency (0-10)")
    cost_w:        float = Field(default=0.3, ge=0, description="Weight for cost (0-10)")
    reliability_w: float = Field(default=0.2, ge=0, description="Weight for reliability (0-10)")
    local_pref_w:  float = Field(default=0.1, ge=0, description="Weight for local preference (0-10)")


@app.get("/")
def root():
    return {"status": "ok", "service": "Conductor AWS LLM Router"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat")
def route_chat(req: ChatRequest):
    total = req.latency_w + req.cost_w + req.reliability_w + req.local_pref_w
    if total == 0:
        raise HTTPException(status_code=400, detail="All weights cannot be zero")

    weights = {
        "latency":     req.latency_w     / total,
        "cost":        req.cost_w        / total,
        "reliability": req.reliability_w / total,
        "local_pref":  req.local_pref_w  / total,
    }

    return chat(req.prompt, weights)


@app.post("/break/{model_name}")
def break_model(model_name: str):
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Available: {list(models.keys())}")
    models[model_name]["latency"]     = 0.01
    models[model_name]["reliability"] = 0.1
    consecutive_drops[model_name]     = 0
    print(f"[CONDUCTOR-AWS] 💥 {model_name} manually degraded for demo.")
    return {"broken": model_name, "message": f"{model_name} degraded — Conductor will reroute on next request."}


@app.post("/restore/{model_name}")
def restore_model(model_name: str):
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
    print(f"[CONDUCTOR-AWS] 🔄 Restoring — re-benchmarking all models...")
    benchmark_all_models()
    return {"restored": model_name, "message": "All models re-benchmarked."}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)