"""
Mock Project — "NeuralNav Assistant"
A simple Flask app that would normally call OpenAI directly.
Instead it calls Conductor's POST /chat endpoint — proving Conductor
is a drop-in replacement for any LLM-powered project.
"""

from flask import Flask, request, jsonify, render_template_string
import requests

app = Flask(__name__)

CONDUCTOR_URL = "http://localhost:8000/chat"

# ── Default weights (can be changed live via the UI) ──────────────────────────
DEFAULT_WEIGHTS = {
    "latency_w":     5,
    "cost_w":        3,
    "reliability_w": 2,
    "local_pref_w":  0,
}

# ── Simple chat UI ─────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>NeuralNav Assistant (powered by Conductor)</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #0f0f0f; color: #eee; }
        h1 { color: #7c3aed; }
        .subtitle { color: #888; margin-top: -10px; font-size: 14px; }
        .chat-box { background: #1a1a1a; border-radius: 10px; padding: 20px; min-height: 300px; margin: 20px 0; }
        .message { margin: 12px 0; }
        .user { color: #60a5fa; }
        .assistant { color: #34d399; }
        .meta { color: #888; font-size: 12px; margin-top: 4px; }
        input[type=text] { width: 100%; padding: 10px; background: #1a1a1a; border: 1px solid #333; color: #eee; border-radius: 6px; font-size: 15px; }
        button { padding: 10px 20px; background: #7c3aed; color: white; border: none; border-radius: 6px; cursor: pointer; margin-top: 8px; }
        button:hover { background: #6d28d9; }
        .weights { background: #1a1a1a; border-radius: 10px; padding: 16px; margin-bottom: 20px; }
        .weights h3 { margin-top: 0; color: #a78bfa; }
        .weight-row { display: flex; align-items: center; gap: 12px; margin: 8px 0; }
        .weight-row label { width: 120px; font-size: 13px; color: #aaa; }
        .weight-row input[type=range] { flex: 1; }
        .weight-row span { width: 20px; text-align: right; font-size: 13px; }
        .badge { display: inline-block; background: #7c3aed; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-left: 8px; }
    </style>
</head>
<body>
    <h1>NeuralNav Assistant</h1>
    <p class="subtitle">Powered by <strong>Conductor</strong> — intelligent LLM routing via AWS Bedrock</p>

    <div class="weights">
        <h3>Routing Weights (change live)</h3>
        <div class="weight-row">
            <label>Latency</label>
            <input type="range" id="latency_w" min="0" max="10" value="5">
            <span id="latency_val">5</span>
        </div>
        <div class="weight-row">
            <label>Cost</label>
            <input type="range" id="cost_w" min="0" max="10" value="3">
            <span id="cost_val">3</span>
        </div>
        <div class="weight-row">
            <label>Reliability</label>
            <input type="range" id="reliability_w" min="0" max="10" value="2">
            <span id="reliability_val">2</span>
        </div>
    </div>

    <div class="chat-box" id="chat"></div>

    <input type="text" id="prompt" placeholder="Ask anything..." onkeydown="if(event.key==='Enter') send()">
    <button onclick="send()">Send</button>

    <script>
        // Update slider labels
        ['latency', 'cost', 'reliability'].forEach(k => {
            document.getElementById(k+'_w').addEventListener('input', function() {
                document.getElementById(k+'_val').textContent = this.value;
            });
        });

        async function send() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) return;

            const weights = {
                latency_w:     parseInt(document.getElementById('latency_w').value),
                cost_w:        parseInt(document.getElementById('cost_w').value),
                reliability_w: parseInt(document.getElementById('reliability_w').value),
                local_pref_w:  0
            };

            // Show user message
            const chat = document.getElementById('chat');
            chat.innerHTML += `<div class="message"><span class="user">You:</span> ${prompt}</div>`;
            document.getElementById('prompt').value = '';

            // Call Conductor
            const res = await fetch('/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt, ...weights})
            });
            const data = await res.json();

            chat.innerHTML += `
                <div class="message">
                    <span class="assistant">Assistant:</span> ${data.response}
                    <div class="meta">
                        Routed to <strong>${data.model_chosen}</strong>
                        <span class="badge">${data.latency}s</span>
                        &nbsp;scores: haiku=${data.scores.haiku} · llama3=${data.scores.llama3} · mistral=${data.scores.mistral}
                    </div>
                </div>`;
            chat.scrollTop = chat.scrollHeight;
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/ask", methods=["POST"])
def ask():
    """
    This is where a normal app would call OpenAI.
    We call Conductor instead — zero other changes needed.
    """
    body = request.json
    prompt = body.get("prompt", "")

    # Pass weights through to Conductor
    payload = {
        "prompt":        prompt,
        "latency_w":     body.get("latency_w",     DEFAULT_WEIGHTS["latency_w"]),
        "cost_w":        body.get("cost_w",         DEFAULT_WEIGHTS["cost_w"]),
        "reliability_w": body.get("reliability_w",  DEFAULT_WEIGHTS["reliability_w"]),
        "local_pref_w":  body.get("local_pref_w",   DEFAULT_WEIGHTS["local_pref_w"]),
    }

    response = requests.post(CONDUCTOR_URL, json=payload)
    return jsonify(response.json())


if __name__ == "__main__":
    print("Mock NeuralNav app running at http://localhost:5001")
    print("Conductor endpoint:", CONDUCTOR_URL)
    app.run(port=5005, debug=True)