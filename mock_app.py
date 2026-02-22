"""
Mock Project — "NeuralNav Assistant"
A simple Flask app that would normally call OpenAI directly.
Instead it calls Conductor's POST /chat endpoint — proving Conductor
is a drop-in replacement for any LLM-powered project.

New features:
- Shared conversation memory across all models (no context loss on model switch)
- If cost weight is high → local model summarizes history (free)
- If cost weight is low  → last 3 turns sent raw (more accurate, slightly pricier)
- Dynamic benchmark cooldown shown in UI
- Failure simulation buttons
"""

from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context
import requests
import subprocess
import json

app = Flask(__name__)

CONDUCTOR_URL = "http://localhost:8000"

DEFAULT_WEIGHTS = {
    "latency_w":     5,
    "cost_w":        3,
    "reliability_w": 2,
    "local_pref_w":  0,
}

# ── Shared conversation memory ─────────────────────────────────────────────────
conversation_history = []


def summarize_with_local(history_text: str) -> str:
    """Use local Ollama model to summarize history — free, no cloud cost."""
    try:
        import ollama
        result = ollama.chat(
            model="qwen2:0.5b",
            messages=[{
                "role": "user",
                "content": f"Summarize this conversation in 2 sentences to use as context:\n{history_text}"
            }]
        )
        return result["message"]["content"]
    except Exception:
        # Ollama not available — fall back to last 4 raw lines
        lines = history_text.strip().split("\n")
        return "\n".join(lines[-4:])


def build_context(cost_w_normalized: float, latency_w_normalized: float):
    if not conversation_history:
        return "", "no_history"
    recent = "\n".join(conversation_history[-6:])
    return f"Here is the recent conversation for context only — do not repeat it:\n{recent}\n\nNow respond to the user's new message: ", "raw_recent"


def process_chat_request(body):
    prompt = body.get("prompt", "")
    latency_w = body.get("latency_w", DEFAULT_WEIGHTS["latency_w"])
    cost_w = body.get("cost_w", DEFAULT_WEIGHTS["cost_w"])
    reliability_w = body.get("reliability_w", DEFAULT_WEIGHTS["reliability_w"])
    local_pref_w = body.get("local_pref_w", DEFAULT_WEIGHTS["local_pref_w"])

    total = latency_w + cost_w + reliability_w + local_pref_w
    cost_w_normalized = cost_w / total if total > 0 else 0
    latency_w_normalized = latency_w / total if total > 0 else 0

    context, context_mode = build_context(cost_w_normalized, latency_w_normalized)
    full_prompt = f"{context}{prompt}" if context else prompt

    conversation_history.append(f"User: {prompt}")

    payload = {
        "prompt": full_prompt,
        "latency_w": latency_w,
        "cost_w": cost_w,
        "reliability_w": reliability_w,
        "local_pref_w": local_pref_w,
    }

    response = requests.post(f"{CONDUCTOR_URL}/chat", json=payload)
    data = response.json()
    data["context_mode"] = context_mode

    conversation_history.append(f"Assistant: {data.get('response', '')}")

    if len(conversation_history) > 20:
        conversation_history.pop(0)
        conversation_history.pop(0)

    return data


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>NeuralNav Assistant (powered by Conductor)</title>
    <style>
        body { font-family: sans-serif; max-width: 860px; margin: 40px auto; padding: 0 20px; background: #0f0f0f; color: #eee; }
        h1 { color: #7c3aed; margin-bottom: 4px; }
        .subtitle { color: #888; margin-top: 0; font-size: 14px; margin-bottom: 20px; }
        .chat-box { background: #1a1a1a; border-radius: 10px; padding: 20px; min-height: 300px; margin: 20px 0; }
        .message { margin: 14px 0; line-height: 1.5; }
        .user { color: #60a5fa; font-weight: bold; }
        .assistant { color: #34d399; font-weight: bold; }
        .meta { color: #888; font-size: 12px; margin-top: 6px; padding: 6px 10px; background: #111; border-radius: 6px; display: inline-block; }
        .model-badge { display: inline-block; background: #7c3aed; color: white; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-right: 6px; }
        .model-badge.fallback { background: #b45309; }
        .latency-badge { display: inline-block; background: #1e3a2f; color: #34d399; padding: 3px 8px; border-radius: 12px; font-size: 11px; margin-right: 4px; }
        .cooldown-badge { display: inline-block; background: #1e2535; color: #60a5fa; padding: 3px 8px; border-radius: 12px; font-size: 11px; }
        input[type=text] { width: 100%; padding: 10px; background: #1a1a1a; border: 1px solid #333; color: #eee; border-radius: 6px; font-size: 15px; box-sizing: border-box; }
        .send-btn { padding: 10px 24px; background: #7c3aed; color: white; border: none; border-radius: 6px; cursor: pointer; margin-top: 8px; font-size: 15px; }
        .send-btn:hover { background: #6d28d9; }
        .panel { background: #1a1a1a; border-radius: 10px; padding: 16px; margin-bottom: 16px; }
        .panel h3 { margin-top: 0; color: #a78bfa; font-size: 14px; }
        .weight-row { display: flex; align-items: center; gap: 12px; margin: 8px 0; }
        .weight-row label { width: 150px; font-size: 13px; color: #aaa; }
        .weight-row input[type=range] { flex: 1; accent-color: #7c3aed; }
        .weight-row span { width: 20px; text-align: right; font-size: 13px; color: #eee; }
        .local-section { margin-top: 14px; padding-top: 14px; border-top: 1px solid #333; }
        .download-btn { padding: 8px 16px; background: #1e3a2f; color: #34d399; border: 1px solid #34d399; border-radius: 6px; cursor: pointer; font-size: 13px; }
        .download-btn:hover { background: #34d399; color: #000; }
        .download-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .dl-status { font-size: 12px; color: #888; margin-left: 10px; }
        #local-row { display: none; }
        .scores { font-size: 11px; color: #555; margin-top: 3px; }
        .memory-tag { font-size: 11px; color: #6366f1; margin-top: 2px; }
        .break-btns { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 6px; }
        .break-btn { padding: 7px 14px; background: #3a1e1e; color: #f87171; border: 1px solid #f87171; border-radius: 6px; cursor: pointer; font-size: 12px; }
        .break-btn:hover { background: #f87171; color: #000; }
        .restore-btn { padding: 7px 14px; background: #1e2a3a; color: #60a5fa; border: 1px solid #60a5fa; border-radius: 6px; cursor: pointer; font-size: 12px; }
        .restore-btn:hover { background: #60a5fa; color: #000; }
        .break-status { font-size: 12px; margin-top: 8px; min-height: 18px; }
    </style>
</head>
<body>
    <h1>NeuralNav Assistant</h1>
    <p class="subtitle">Powered by <strong>Conductor</strong> — intelligent LLM routing via AWS Bedrock + Strands Agents</p>

    <!-- Routing weights -->
    <div class="panel">
        <h3>⚡ Routing Weights (drag to switch models live)</h3>
        <div class="weight-row">
            <label>🚀 Latency</label>
            <input type="range" id="latency_w" min="0" max="10" value="5">
            <span id="latency_val">5</span>
        </div>
        <div class="weight-row">
            <label>💰 Cost</label>
            <input type="range" id="cost_w" min="0" max="10" value="3">
            <span id="cost_val">3</span>
        </div>
        <div class="weight-row">
            <label>🛡 Reliability</label>
            <input type="range" id="reliability_w" min="0" max="10" value="2">
            <span id="reliability_val">2</span>
        </div>

        <div class="local-section">
            <button class="download-btn" id="dl-btn" onclick="downloadLocal()">⬇ Add Local Model (qwen2:0.5b)</button>
            <span class="dl-status" id="dl-status"></span>
            <div class="weight-row" id="local-row" style="margin-top:10px;">
                <label>🖥 Local Preference</label>
                <input type="range" id="local_pref_w" min="0" max="10" value="5">
                <span id="local_val">5</span>
            </div>
        </div>
    </div>

    <!-- Failure simulation -->
    <div class="panel">
        <h3>💥 Simulate Model Failure (demo)</h3>
        <div class="break-btns">
            <button class="break-btn" onclick="breakModel('haiku')">💥 Break Haiku</button>
            <button class="break-btn" onclick="breakModel('llama3')">💥 Break Llama3</button>
            <button class="break-btn" onclick="breakModel('mistral')">💥 Break Mistral</button>
            <button class="restore-btn" onclick="restoreAll()">🔄 Restore All</button>
        </div>
        <div class="break-status" id="break-status"></div>
    </div>

    <div class="chat-box" id="chat"></div>

    <input type="text" id="prompt" placeholder="Ask anything..." onkeydown="if(event.key==='Enter') send()">
    <button class="send-btn" onclick="send()">Send →</button>

    <script>
        ['latency', 'cost', 'reliability', 'local_pref'].forEach(k => {
            const el = document.getElementById(k + '_w');
            if (el) el.addEventListener('input', function() {
                document.getElementById(k + '_val').textContent = this.value;
            });
        });

        async function downloadLocal() {
            const btn = document.getElementById('dl-btn');
            const status = document.getElementById('dl-status');
            btn.disabled = true;
            status.textContent = 'Pulling qwen2:0.5b... (may take a minute)';
            const res = await fetch('/download_local', { method: 'POST' });
            const data = await res.json();
            if (data.success) {
                status.textContent = '✅ Local model ready!';
                document.getElementById('local-row').style.display = 'flex';
                btn.textContent = '✅ Local Model Active';
            } else {
                status.textContent = '❌ Failed: ' + data.error;
                btn.disabled = false;
            }
        }

        async function breakModel(name) {
            const status = document.getElementById('break-status');
            status.style.color = '#f87171';
            status.textContent = `💥 Breaking ${name}...`;
            await fetch('/break/' + name, { method: 'POST' });
            status.textContent = `⚠️ ${name} degraded! Send a message to watch Conductor reroute automatically.`;
        }

        async function restoreAll() {
            const status = document.getElementById('break-status');
            status.style.color = '#60a5fa';
            status.textContent = '🔄 Re-benchmarking all models...';
            await fetch('/restore/all', { method: 'POST' });
            status.textContent = '✅ All models restored and re-benchmarked!';
        }

        async function send() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) return;

            const localRow = document.getElementById('local-row');
            const localActive = localRow.style.display !== 'none';
            const costW = parseInt(document.getElementById('cost_w').value);

            const weights = {
                latency_w:     parseInt(document.getElementById('latency_w').value),
                cost_w:        costW,
                reliability_w: parseInt(document.getElementById('reliability_w').value),
                local_pref_w:  localActive ? parseInt(document.getElementById('local_pref_w').value) : 0,
            };

            const chat = document.getElementById('chat');
            chat.innerHTML += `<div class="message"><span class="user">You:</span> ${prompt}</div>`;
            document.getElementById('prompt').value = '';
            chat.innerHTML += `<div class="message" id="thinking"><span style="color:#444">⏳ Conductor routing...</span></div>`;
            chat.scrollTop = chat.scrollHeight;

            const params = new URLSearchParams({ prompt, ...weights });
            const evt = new EventSource('/ask_stream?' + params.toString());

            evt.addEventListener('status', (event) => {
                const status = JSON.parse(event.data);
                const thinkEl = document.getElementById('thinking');
                if (thinkEl) {
                    thinkEl.innerHTML = `<span style="color:#666">⏳ ${status.message}</span>`;
                }
            });

            evt.addEventListener('final', (event) => {
                const data = JSON.parse(event.data);
                const thinkEl = document.getElementById('thinking');
                if (thinkEl) thinkEl.remove();

                const isFallback = data.model_chosen && data.model_chosen.includes('fallback');
                const badgeClass = isFallback ? 'model-badge fallback' : 'model-badge';
                const scoreStr = Object.entries(data.scores || {})
                    .map(([k, v]) => `${k}: ${v}`)
                    .join(' · ');

                let memoryMode = '☁️ memory: raw context (accurate)';
                if (data.context_mode === 'local_summary') memoryMode = '🖥 memory: local summarization (free)';
                if (data.context_mode === 'latency_fast_path') memoryMode = '⚡ memory: skipped for speed';

                chat.innerHTML += `
                    <div class="message">
                        <span class="assistant">Assistant:</span> ${data.response}
                        <div class="meta">
                            <span class="${badgeClass}">→ ${data.model_chosen}</span>
                            <span class="latency-badge">${data.latency}s</span>
                            <span class="cooldown-badge">⏱ benchmark every ${data.benchmark_cooldown}s</span>
                            <div class="scores">scores: ${scoreStr}</div>
                            <div class="memory-tag">${memoryMode}</div>
                        </div>
                    </div>`;
                chat.scrollTop = chat.scrollHeight;

                if (data.ollama_available && localRow.style.display === 'none') {
                    localRow.style.display = 'flex';
                    document.getElementById('dl-status').textContent = '✅ Local model detected!';
                    document.getElementById('dl-btn').textContent = '✅ Local Model Active';
                    document.getElementById('dl-btn').disabled = true;
                }

                evt.close();
            });

            evt.addEventListener('error', () => {
                const thinkEl = document.getElementById('thinking');
                if (thinkEl) thinkEl.remove();
                chat.innerHTML += `<div class="message"><span style="color:#f87171">Request failed.</span></div>`;
                evt.close();
            });
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/download_local", methods=["POST"])
def download_local():
    try:
        result = subprocess.run(
            ["ollama", "pull", "qwen2:0.5b"],
            timeout=300
        )
        if result.returncode == 0:
            requests.post(f"{CONDUCTOR_URL}/enable_local")
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "ollama pull failed"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    

@app.route("/break/<model_name>", methods=["POST"])
def break_model(model_name):
    res = requests.post(f"{CONDUCTOR_URL}/break/{model_name}")
    return jsonify(res.json())


@app.route("/restore/all", methods=["POST"])
def restore_all():
    res = requests.post(f"{CONDUCTOR_URL}/restore/haiku")
    return jsonify(res.json())


@app.route("/ask", methods=["POST"])
def ask():
    return jsonify(process_chat_request(request.json))


@app.route("/ask_stream", methods=["GET"])
def ask_stream():
    body = {
        "prompt": request.args.get("prompt", ""),
        "latency_w": int(request.args.get("latency_w", DEFAULT_WEIGHTS["latency_w"])),
        "cost_w": int(request.args.get("cost_w", DEFAULT_WEIGHTS["cost_w"])),
        "reliability_w": int(request.args.get("reliability_w", DEFAULT_WEIGHTS["reliability_w"])),
        "local_pref_w": int(request.args.get("local_pref_w", DEFAULT_WEIGHTS["local_pref_w"])),
    }

    @stream_with_context
    def generate():
        yield f"event: status\ndata: {json.dumps({'message': 'Routing request...'})}\n\n"
        yield f"event: status\ndata: {json.dumps({'message': 'Calling selected model...'})}\n\n"
        try:
            data = process_chat_request(body)
            yield f"event: final\ndata: {json.dumps(data)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    print("Mock NeuralNav app running at http://localhost:5005")
    print("Conductor endpoint:", CONDUCTOR_URL)
    app.run(port=5005, debug=True)
