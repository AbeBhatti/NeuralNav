"""
Conductor AWS — FastAPI endpoint
POST /chat → scores Bedrock models, routes to winner, returns response + metadata
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn

from router_agent import benchmark_all_models, chat


# ── Startup: benchmark all models before accepting requests ───────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[CONDUCTOR-AWS] Starting up — running initial benchmark...")
    benchmark_all_models()
    print("[CONDUCTOR-AWS] Ready to route requests.")
    yield

app = FastAPI(
    title="Conductor AWS",
    description="Intelligent LLM router using AWS Bedrock models",
    lifespan=lifespan
)


# ── Request / Response schemas ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    prompt:        str   = Field(..., description="The user prompt to route")
    latency_w:     float = Field(default=0.4, ge=0, description="Weight for latency (0-10)")
    cost_w:        float = Field(default=0.3, ge=0, description="Weight for cost (0-10)")
    reliability_w: float = Field(default=0.2, ge=0, description="Weight for reliability (0-10)")
    local_pref_w:  float = Field(default=0.1, ge=0, description="Weight for local preference (0-10)")


class ChatResponse(BaseModel):
    response:     str
    model_chosen: str
    scores:       dict
    latency:      float
    tokens:       dict


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "Conductor AWS LLM Router"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def route_chat(req: ChatRequest):
    # Normalize weights so they always sum to 1
    total = req.latency_w + req.cost_w + req.reliability_w + req.local_pref_w
    if total == 0:
        raise HTTPException(status_code=400, detail="All weights cannot be zero")

    weights = {
        "latency":    req.latency_w     / total,
        "cost":       req.cost_w        / total,
        "reliability":req.reliability_w / total,
        "local_pref": req.local_pref_w  / total,
    }

    result = chat(req.prompt, weights)
    return result


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)