import os
os.environ["DD_TRACE_ENABLED"] = "false"
os.environ["DD_TRACE_AGENT_URL"] = "http://localhost:1"

from dotenv import load_dotenv
load_dotenv()

import time
import json
import boto3
from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import llm
from datadog import initialize, statsd
from strands import Agent
from strands.models.bedrock import BedrockModel

LLMObs.enable(
    ml_app="conductor",
    api_key=os.getenv("DD_API_KEY"),
    site=os.getenv("DD_SITE", "datadoghq.com"),
    agentless_enabled=True
)

initialize(
    api_key=os.getenv("DD_API_KEY"),
    app_key=os.getenv("DD_APP_KEY", ""),
    host_name="conductor-aws"
)

statsd.host = "localhost"
statsd.port = 8125

bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-west-2"))

MODEL_IDS = {
    "haiku":   "anthropic.claude-3-haiku-20240307-v1:0",
    "llama3":  "meta.llama3-8b-instruct-v1:0",
    "mistral": "mistral.mistral-7b-instruct-v0:2",
}

MODEL_PRICING = {
    "haiku":   0.25,
    "llama3":  0.30,
    "mistral": 0.15,
}

IS_LOCAL = {"haiku": False, "llama3": False, "mistral": False}

max_cost_per_call = 0.001

models = {
    name: {
        "latency": None, "cost": None, "reliability": None,
        "input_tokens": None, "output_tokens": None, "tokens_per_sec": None,
        "error_rate": 0.0, "total_calls": 0, "total_errors": 0
    }
    for name in MODEL_IDS
}

initial_scores    = {}
consecutive_drops = {}
last_benchmark_time = 0


def get_generation_config(weights: dict) -> dict:
    latency_w = weights.get("latency", 0.0)
    cost_w = weights.get("cost", 0.0)

    if latency_w >= 0.5:
        return {"max_tokens": 160, "temperature": 0.2}
    if latency_w >= 0.35:
        return {"max_tokens": 256, "temperature": 0.4}
    if cost_w >= 0.45:
        return {"max_tokens": 220, "temperature": 0.5}
    return {"max_tokens": 512, "temperature": 0.7}


def get_benchmark_cooldown(weights):
    """
    Dynamic cooldown based on cost weight.
    High cost priority → benchmark less often (save API calls).
    Low cost priority  → benchmark aggressively (always fresh data).

    Configurable per project type — e.g.:
      - Fintech / cost-sensitive: set BASE=120, MAXIMUM=600
      - Trading / latency-critical: set BASE=15, MAXIMUM=60
    """
    BASE    = 20   # seconds minimum  (kept low for demo)
    MAXIMUM = 90   # seconds maximum  (kept low for demo)
    cost_w  = weights.get("cost", 0)  # already normalized 0-1
    cooldown = BASE + (cost_w * (MAXIMUM - BASE))
    return cooldown


def calculate_cost_score(model_name, input_tokens, output_tokens):
    global max_cost_per_call
    price_per_million = MODEL_PRICING[model_name]
    total_tokens = (input_tokens or 0) + (output_tokens or 0)
    cost_usd = (total_tokens / 1_000_000) * price_per_million
    if cost_usd > max_cost_per_call:
        max_cost_per_call = cost_usd
    if max_cost_per_call == 0:
        return 1.0
    return round(1.0 - (cost_usd / max_cost_per_call), 3)


def report_to_datadog(model_name, latency, input_tokens, output_tokens, cost_usd, tokens_per_sec, success, chosen=False):
    tags = [f"model:{model_name}", "ml_app:conductor", "provider:bedrock"]
    try:
        statsd.gauge("conductor.latency",        latency,             tags=tags)
        statsd.gauge("conductor.tokens.input",   input_tokens or 0,   tags=tags)
        statsd.gauge("conductor.tokens.output",  output_tokens or 0,  tags=tags)
        statsd.gauge("conductor.tokens.per_sec", tokens_per_sec or 0, tags=tags)
        statsd.gauge("conductor.cost_usd",       cost_usd,            tags=tags)
        statsd.increment("conductor.calls",                            tags=tags)
        if not success:
            statsd.increment("conductor.errors",                       tags=tags)
        if chosen:
            statsd.increment("conductor.chosen",                       tags=tags)
    except Exception:
        pass


def score_models(weights):
    scores = {}
    for name, metrics in models.items():
        if metrics["latency"] is None or metrics["reliability"] is None or metrics["cost"] is None:
            scores[name] = 0
            continue
        local_bonus = 0.3 if IS_LOCAL[name] else 0.0
        score = (
            metrics["latency"]     * weights["latency"] +
            metrics["cost"]        * weights["cost"] +
            metrics["reliability"] * weights["reliability"] +
            local_bonus            * weights.get("local_pref", 0)
        )
        scores[name] = score
    return max(scores, key=scores.get), scores


def _invoke_haiku(prompt: str, max_tokens: int = 512, temperature: float = 0.7):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}]
    })
    start   = time.time()
    resp    = bedrock.invoke_model(modelId=MODEL_IDS["haiku"], body=body)
    latency = time.time() - start
    out     = json.loads(resp["body"].read())
    text    = out["content"][0]["text"]
    inp_tok = out["usage"]["input_tokens"]
    out_tok = out["usage"]["output_tokens"]
    tps     = out_tok / latency if latency > 0 else 0
    return text, latency, inp_tok, out_tok, tps


def _invoke_llama3(prompt: str, max_tokens: int = 512, temperature: float = 0.7):
    body = json.dumps({
        "prompt": f"<|begin_of_text|><|user|>\n{prompt}\n<|assistant|>\n",
        "max_gen_len": max_tokens,
        "temperature": temperature,
    })
    start   = time.time()
    resp    = bedrock.invoke_model(modelId=MODEL_IDS["llama3"], body=body)
    latency = time.time() - start
    out     = json.loads(resp["body"].read())
    text    = out["generation"]
    inp_tok = len(prompt) // 4
    out_tok = len(text) // 4
    tps     = out_tok / latency if latency > 0 else 0
    return text, latency, inp_tok, out_tok, tps


def _invoke_mistral(prompt: str, max_tokens: int = 512, temperature: float = 0.7):
    body = json.dumps({
        "prompt": f"<s>[INST] {prompt} [/INST]",
        "max_tokens": max_tokens,
        "temperature": temperature,
    })
    start   = time.time()
    resp    = bedrock.invoke_model(modelId=MODEL_IDS["mistral"], body=body)
    latency = time.time() - start
    out     = json.loads(resp["body"].read())
    text    = out["outputs"][0]["text"]
    inp_tok = len(prompt) // 4
    out_tok = len(text) // 4
    tps     = out_tok / latency if latency > 0 else 0
    return text, latency, inp_tok, out_tok, tps


@llm(model_provider="bedrock", model_name="claude-3-haiku")
def call_haiku(prompt, max_tokens: int = 512, temperature: float = 0.7):
    return _invoke_haiku(prompt, max_tokens=max_tokens, temperature=temperature)

@llm(model_provider="bedrock", model_name="llama3-8b")
def call_llama3(prompt, max_tokens: int = 512, temperature: float = 0.7):
    return _invoke_llama3(prompt, max_tokens=max_tokens, temperature=temperature)

@llm(model_provider="bedrock", model_name="mistral-7b")
def call_mistral(prompt, max_tokens: int = 512, temperature: float = 0.7):
    return _invoke_mistral(prompt, max_tokens=max_tokens, temperature=temperature)

CALLERS = {"haiku": call_haiku, "llama3": call_llama3, "mistral": call_mistral}

# ── Optional local Ollama model ────────────────────────────────────────────────
OLLAMA_AVAILABLE = False
try:
    import ollama as ollama_client
    ollama_client.chat(model="qwen2:0.5b", messages=[{"role": "user", "content": "hi"}])

    MODEL_IDS["local"]     = "qwen2:0.5b"
    MODEL_PRICING["local"] = 0.0
    IS_LOCAL["local"]      = True
    models["local"] = {
        "latency": None, "cost": None, "reliability": None,
        "input_tokens": None, "output_tokens": None, "tokens_per_sec": None,
        "error_rate": 0.0, "total_calls": 0, "total_errors": 0
    }

    @llm(model_provider="ollama", model_name="qwen2:0.5b")
    def call_local(prompt, max_tokens: int = 512, temperature: float = 0.7):
        start    = time.time()
        response = ollama_client.chat(
            model="qwen2:0.5b",
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": max_tokens, "temperature": temperature}
        )
        latency  = time.time() - start
        inp_tok  = response.get("prompt_eval_count", 0)
        out_tok  = response.get("eval_count", 0)
        tps      = out_tok / latency if latency > 0 else 0
        return response["message"]["content"], latency, inp_tok, out_tok, tps

    CALLERS["local"] = call_local
    OLLAMA_AVAILABLE = True
    print("[CONDUCTOR-AWS] Ollama detected — local model (qwen2:0.5b) added to pool.")
except Exception:
    print("[CONDUCTOR-AWS] Ollama not available — running cloud-only (3 Bedrock models).")


# ── Strands Agent ──────────────────────────────────────────────────────────────
strands_agent = Agent(
    model=BedrockModel(
        model_id=MODEL_IDS["haiku"],
        region_name=os.getenv("AWS_REGION", "us-west-2")
    ),
    system_prompt="You are Conductor, an intelligent LLM router that selects the best model based on performance metrics."
)


def benchmark_all_models():
    global last_benchmark_time
    print("\n[CONDUCTOR-AWS] Benchmarking models...")
    probe = "say hi in one word"

    for name in models:
        try:
            _, latency, inp, out, tps = CALLERS[name](probe, max_tokens=16, temperature=0.1)
            latency_score = 1 / (latency + 0.01)
            cost_score    = calculate_cost_score(name, inp, out)
            cost_usd      = ((inp + out) / 1_000_000) * MODEL_PRICING[name]

            models[name]["latency"]        = round(latency_score, 3)
            models[name]["cost"]           = cost_score
            models[name]["reliability"]    = 1.0
            models[name]["input_tokens"]   = inp
            models[name]["output_tokens"]  = out
            models[name]["tokens_per_sec"] = round(tps, 1)

            report_to_datadog(name, latency, inp, out, cost_usd, tps, success=True)
            print(f"  {name}: latency={latency:.2f}s | {inp}in/{out}out tok | {tps:.0f} tok/s | cost_score={cost_score:.3f}")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")
            models[name].update({"latency": 0.0, "cost": 0.0, "reliability": 0.0})
            report_to_datadog(name, 99, 0, 0, 0, 0, success=False)

    last_benchmark_time = time.time()
    for name, metrics in models.items():
        initial_scores[name]    = metrics["latency"] or 0
        consecutive_drops[name] = 0

    print("[CONDUCTOR-AWS] Benchmark complete.\n")


def update_metrics(model_name, latency, success, inp, out, tps, weights):
    global last_benchmark_time

    old_latency   = models[model_name]["latency"] or 0
    latency_score = min(1 / (latency + 0.1), 1.0)
    models[model_name]["latency"] = round(0.7 * old_latency + 0.3 * latency_score, 3)

    cost_score = calculate_cost_score(model_name, inp, out)
    cost_usd   = ((inp + out) / 1_000_000) * MODEL_PRICING[model_name]
    models[model_name]["cost"]           = cost_score
    models[model_name]["input_tokens"]   = inp
    models[model_name]["output_tokens"]  = out
    models[model_name]["tokens_per_sec"] = round(tps, 1)

    models[model_name]["total_calls"] += 1
    if not success:
        models[model_name]["total_errors"] += 1
    models[model_name]["error_rate"] = round(
        models[model_name]["total_errors"] / models[model_name]["total_calls"], 3
    )

    old_rel = models[model_name]["reliability"] or 1.0
    models[model_name]["reliability"] = round(
        0.8 * old_rel + 0.2 * (1.0 if success else 0.0), 3
    )

    report_to_datadog(model_name, latency, inp, out, cost_usd, tps, success, chosen=True)

    current_score = (
        models[model_name]["latency"]     * weights["latency"] +
        models[model_name]["cost"]        * weights["cost"] +
        models[model_name]["reliability"] * weights["reliability"]
    )
    best_initial = max(initial_scores.values()) if initial_scores else 0

    if current_score < best_initial:
        consecutive_drops[model_name] = consecutive_drops.get(model_name, 0) + 1
    else:
        consecutive_drops[model_name] = 0

    # Dynamic cooldown — high cost weight = benchmark less often
    cooldown = get_benchmark_cooldown(weights)
    cooldown_passed = (time.time() - last_benchmark_time) > cooldown

    if consecutive_drops.get(model_name, 0) >= 2 and cooldown_passed:
        print(f"\n[CONDUCTOR-AWS] {model_name} degraded. Cooldown was {cooldown:.0f}s. Re-benchmarking...")
        benchmark_all_models()


def chat(prompt: str, weights: dict) -> dict:
    chosen, scores = score_models(weights)
    gen_cfg = get_generation_config(weights)
    print(f"[CONDUCTOR-AWS] Routing to → {chosen.upper()} (cooldown={get_benchmark_cooldown(weights):.0f}s)")

    try:
        response, latency, inp, out, tps = CALLERS[chosen](
            prompt,
            max_tokens=gen_cfg["max_tokens"],
            temperature=gen_cfg["temperature"]
        )
        update_metrics(chosen, latency, True, inp, out, tps, weights)
        return {
            "response":         response,
            "model_chosen":     chosen,
            "scores":           {k: round(v, 3) for k, v in scores.items()},
            "latency":          round(latency, 3),
            "tokens":           {"input": inp, "output": out},
            "ollama_available": OLLAMA_AVAILABLE,
            "benchmark_cooldown": round(get_benchmark_cooldown(weights)),
            "generation":       gen_cfg,
        }

    except Exception as e:
        print(f"[CONDUCTOR-AWS] {chosen} failed: {e}. Falling back...")
        update_metrics(chosen, 99, False, 0, 0, 0, weights)
        fallback = "mistral" if chosen != "mistral" else "haiku"
        response, latency, inp, out, tps = CALLERS[fallback](
            prompt,
            max_tokens=gen_cfg["max_tokens"],
            temperature=gen_cfg["temperature"]
        )
        update_metrics(fallback, latency, True, inp, out, tps, weights)
        return {
            "response":         response,
            "model_chosen":     f"{fallback} (fallback from {chosen})",
            "scores":           {k: round(v, 3) for k, v in scores.items()},
            "latency":          round(latency, 3),
            "tokens":           {"input": inp, "output": out},
            "ollama_available": OLLAMA_AVAILABLE,
            "benchmark_cooldown": round(get_benchmark_cooldown(weights)),
            "generation":       gen_cfg,
        }
