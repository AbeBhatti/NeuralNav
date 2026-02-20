import os
os.environ["DD_TRACE_ENABLED"] = "false"
os.environ["DD_TRACE_AGENT_URL"] = "http://localhost:1"

from dotenv import load_dotenv
load_dotenv()

import time
from groq import Groq
import cohere
import ollama
from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import llm
from datadog import initialize, statsd

LLMObs.enable(
    ml_app="conductor",
    api_key=os.getenv("DD_API_KEY"),
    site=os.getenv("DD_SITE", "datadoghq.com"),
    agentless_enabled=True
)

initialize(
    api_key=os.getenv("DD_API_KEY"),
    app_key=os.getenv("DD_APP_KEY", ""),
    host_name="conductor-local"
)

from datadog import statsd
statsd.host = "localhost"
statsd.port = 8125

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
cohere_client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

MODEL_PRICING = {
    "groq":   0.10,
    "cohere": 0.15,
    "local":  0.0
}

max_cost_per_call = 0.001

models = {
    "groq": {
        "latency": None, "cost": None, "reliability": None,
        "input_tokens": None, "output_tokens": None, "tokens_per_sec": None,
        "error_rate": 0.0, "total_calls": 0, "total_errors": 0
    },
    "cohere": {
        "latency": None, "cost": None, "reliability": None,
        "input_tokens": None, "output_tokens": None, "tokens_per_sec": None,
        "error_rate": 0.0, "total_calls": 0, "total_errors": 0
    },
    "local": {
        "latency": None, "cost": 1.0, "reliability": None,
        "input_tokens": None, "output_tokens": None, "tokens_per_sec": None,
        "error_rate": 0.0, "total_calls": 0, "total_errors": 0
    }
}

initial_scores = {}
consecutive_drops = {}
REBENCHMARK_COOLDOWN = 60
last_benchmark_time = 0


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
    tags = [f"model:{model_name}", "ml_app:conductor"]
    try:
        statsd.gauge("conductor.latency", latency, tags=tags)
        statsd.gauge("conductor.tokens.input", input_tokens or 0, tags=tags)
        statsd.gauge("conductor.tokens.output", output_tokens or 0, tags=tags)
        statsd.gauge("conductor.tokens.per_sec", tokens_per_sec or 0, tags=tags)
        statsd.gauge("conductor.cost_usd", cost_usd, tags=tags)
        statsd.increment("conductor.calls", tags=tags)
        if not success:
            statsd.increment("conductor.errors", tags=tags)
        if chosen:
            statsd.increment("conductor.chosen", tags=tags)
    except Exception:
        pass


def get_weights():
    print("\nRate each metric 1-10 (higher = more important):")
    while True:
        try:
            latency = int(input("Latency (response speed): "))
            cost = int(input("Cost efficiency (cheaper = better): "))
            reliability = int(input("Reliability (uptime/errors): "))
            local_pref = int(input("Prefer local model (keeps compute on your machine): "))
            break
        except ValueError:
            print("Please enter numbers only, try again.")
    total = latency + cost + reliability + local_pref
    return {
        "latency": latency / total,
        "cost": cost / total,
        "reliability": reliability / total,
        "local_pref": local_pref / total
    }


def score_models(weights):
    scores = {}
    for name, metrics in models.items():
        if metrics["latency"] is None or metrics["reliability"] is None or metrics["cost"] is None:
            scores[name] = 0
            print(f"  {name}: not benchmarked yet")
            continue
        local_bonus = weights["local_pref"] if name == "local" else 0.0
        score = (
            metrics["latency"] * weights["latency"] +
            metrics["cost"] * weights["cost"] +
            metrics["reliability"] * weights["reliability"] +
            local_bonus * weights["local_pref"]
        )
        scores[name] = score
        print(f"  {name}: {score:.3f} (latency={metrics['latency']:.3f}, cost={metrics['cost']:.3f}, reliability={metrics['reliability']:.3f}, local_bonus={local_bonus * weights['local_pref']:.3f})")
    return max(scores, key=scores.get)


@llm(model_provider="groq", model_name="llama-3.1-8b-instant")
def call_groq(prompt):
    start = time.time()
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    latency = time.time() - start
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    tokens_per_sec = output_tokens / latency if latency > 0 else 0


    return response.choices[0].message.content, latency, input_tokens, output_tokens, tokens_per_sec


@llm(model_provider="cohere", model_name="command-a-03-2025")
def call_cohere(prompt):
    start = time.time()
    response = cohere_client.chat(
        model="command-a-03-2025",
        message=prompt
    )
    latency = time.time() - start
    input_tokens = response.meta.tokens.input_tokens
    output_tokens = response.meta.tokens.output_tokens
    tokens_per_sec = output_tokens / latency if latency > 0 else 0


    return response.text, latency, input_tokens, output_tokens, tokens_per_sec


@llm(model_provider="ollama", model_name="qwen2:0.5b")
def call_local(prompt):
    start = time.time()
    response = ollama.chat(
        model="qwen2:0.5b",
        messages=[{"role": "user", "content": prompt}]
    )
    latency = time.time() - start
    input_tokens = response.get("prompt_eval_count", 0)
    output_tokens = response.get("eval_count", 0)
    tokens_per_sec = output_tokens / latency if latency > 0 else 0


    return response["message"]["content"], latency, input_tokens, output_tokens, tokens_per_sec


def benchmark_all_models():
    global last_benchmark_time
    print("\n[CONDUCTOR] Benchmarking all models...")
    probe = "say hi in one word"

    for name in models:
        try:
            if name == "groq":
                _, latency, inp, out, tps = call_groq(probe)
            elif name == "cohere":
                _, latency, inp, out, tps = call_cohere(probe)
            else:
                _, latency, inp, out, tps = call_local(probe)

            latency_score = 1 / (latency + 0.01)
            cost_score = calculate_cost_score(name, inp, out)
            cost_usd = ((inp + out) / 1_000_000) * MODEL_PRICING[name]

            models[name]["latency"] = round(latency_score, 3)
            models[name]["cost"] = cost_score
            models[name]["reliability"] = 1.0
            models[name]["input_tokens"] = inp
            models[name]["output_tokens"] = out
            models[name]["tokens_per_sec"] = round(tps, 1)

            report_to_datadog(name, latency, inp, out, cost_usd, tps, success=True)
            print(f"  {name}: latency={latency:.2f}s | tokens={inp}in/{out}out | {tps:.0f} tok/s | cost_score={cost_score:.3f} | cost=${cost_usd:.6f}")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")
            models[name]["latency"] = 0.0
            models[name]["cost"] = 0.0
            models[name]["reliability"] = 0.0
            report_to_datadog(name, 99, 0, 0, 0, 0, success=False)

    last_benchmark_time = time.time()
    for name, metrics in models.items():
        initial_scores[name] = metrics["latency"] or 0
        consecutive_drops[name] = 0

    print("[CONDUCTOR] Benchmark complete.\n")


def update_metrics(model_name, latency, success, inp, out, tps, weights):
    global last_benchmark_time

    old_latency = models[model_name]["latency"]
    latency_score = min(1 / (latency + 0.1), 1.0)
    models[model_name]["latency"] = round(0.7 * old_latency + 0.3 * latency_score, 3)

    cost_score = calculate_cost_score(model_name, inp, out)
    cost_usd = ((inp + out) / 1_000_000) * MODEL_PRICING[model_name]
    models[model_name]["cost"] = cost_score
    models[model_name]["input_tokens"] = inp
    models[model_name]["output_tokens"] = out
    models[model_name]["tokens_per_sec"] = round(tps, 1)

    models[model_name]["total_calls"] += 1
    if not success:
        models[model_name]["total_errors"] += 1
    models[model_name]["error_rate"] = round(
        models[model_name]["total_errors"] / models[model_name]["total_calls"], 3
    )
    old_reliability = models[model_name]["reliability"]
    models[model_name]["reliability"] = round(0.8 * old_reliability + 0.2 * (1.0 if success else 0.0), 3)

    report_to_datadog(model_name, latency, inp, out, cost_usd, tps, success, chosen=True)

    print(f"[CONDUCTOR] Updated {model_name}:")
    print(f"  latency={models[model_name]['latency']:.3f} | cost={models[model_name]['cost']:.3f} | reliability={models[model_name]['reliability']:.3f}")
    print(f"  tokens={inp}in/{out}out | {tps:.0f} tok/s | cost=${cost_usd:.6f} | error_rate={models[model_name]['error_rate']:.3f}")

    current_score = (
        models[model_name]["latency"] * weights["latency"] +
        models[model_name]["cost"] * weights["cost"] +
        models[model_name]["reliability"] * weights["reliability"]
    )
    best_initial = max(initial_scores.values())

    if current_score < best_initial:
        consecutive_drops[model_name] = consecutive_drops.get(model_name, 0) + 1
    else:
        consecutive_drops[model_name] = 0

    cooldown_passed = (time.time() - last_benchmark_time) > REBENCHMARK_COOLDOWN
    if consecutive_drops.get(model_name, 0) >= 2 and cooldown_passed:
        print(f"\n[CONDUCTOR] {model_name} degraded. Re-benchmarking all models...")
        benchmark_all_models()


def chat(prompt, weights):
    print("\n[CONDUCTOR] Scoring models...")
    chosen = score_models(weights)
    print(f"[CONDUCTOR] Routing to → {chosen.upper()}")

    try:
        if chosen == "groq":
            response, latency, inp, out, tps = call_groq(prompt)
        elif chosen == "cohere":
            response, latency, inp, out, tps = call_cohere(prompt)
        else:
            response, latency, inp, out, tps = call_local(prompt)

        print(f"[CONDUCTOR] Latency: {latency:.2f}s | Tokens: {inp}in/{out}out | {tps:.0f} tok/s")
        update_metrics(chosen, latency, True, inp, out, tps, weights)
        return response

    except Exception as e:
        print(f"[CONDUCTOR] {chosen} failed: {e}. Falling back to local...")
        update_metrics(chosen, 99, False, 0, 0, 0, weights)
        response, latency, inp, out, tps = call_local(prompt)
        return response


if __name__ == "__main__":
    print("=" * 50)
    print("         CONDUCTOR — LLM Router")
    print("=" * 50)

    benchmark_all_models()

    weights = get_weights()
    print("\n[CONDUCTOR] Weights set. Starting session...\n")

    while True:
        prompt = input("You: ")
        if prompt.lower() in ["quit", "exit", "q"]:
            print("\n[CONDUCTOR] Session ended.")
            break
        response = chat(prompt, weights)
        print(f"\nAssistant: {response}\n")
        print("-" * 50)