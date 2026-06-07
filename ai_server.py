#!/usr/bin/env python3
"""
Le Potato AI Offload Server — runs on your Linux PC
=====================================================
The Le Potato's server.py sends frames here instead of running Ollama locally.
This machine does all the heavy AI lifting and returns the same JSON the
dashboard expects, so nothing else in the stack needs to change.

Architecture:
  [Browser dashboard] ──HTTPS──> [Le Potato :5000]
                                       │
                              /api/think (HTTP, LAN)
                                       │
                                       ▼
                          [This PC :11435] ──localhost──> [Ollama :11434]

Setup (this PC):
  1. Install Ollama:
       curl -fsSL https://ollama.com/install.sh | sh
  2. Pull models:
       ollama pull llama3.2:3b
       ollama pull moondream
  3. Install Python deps:
       pip install flask requests
  4. Run this server:
       python3 ai_server.py
     Or with debug output:
       python3 ai_server.py --debug

Setup (Le Potato) — edit server.py, change one line at the top:
       OFFLOAD_URL = "http://<THIS_PC_LAN_IP>:11435"

Then on the Le Potato run as normal:
       sudo python3 server.py
"""

import json
import os
import re
import sys
import time
import queue
import threading
import traceback
import requests as req

from flask import Flask, request, jsonify, Response

# ═══════════════════════════════════════════════════════════════════════════════
# ── CONFIG — edit these ───────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

LISTEN_HOST    = "0.0.0.0"
LISTEN_PORT    = 11435          # Le Potato will connect to this port

OLLAMA_URL     = "http://localhost:11434"
VISION_MODEL   = "moondream"
TEXT_MODEL     = "llama3.2:3b"
OLLAMA_TIMEOUT = 120            # seconds — plenty of headroom on a real PC

# Optional: restrict which IPs can call this server (Le Potato's LAN IP).
# Set to None to allow any machine on the LAN.
ALLOWED_HOSTS  = None           # e.g. {"192.168.68.127"}

# ═══════════════════════════════════════════════════════════════════════════════
# ── DEBUG ─────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

DEBUG = '--debug' in sys.argv or os.environ.get('LEPOTATO_DEBUG', '0') == '1'

_C = {
    'reset':  '\033[0m',  'grey':   '\033[90m',
    'cyan':   '\033[96m', 'green':  '\033[92m',
    'yellow': '\033[93m', 'red':    '\033[91m',
    'purple': '\033[95m', 'bold':   '\033[1m',
}
_LEVEL_COLOR = {
    'INFO': 'cyan', 'OK': 'green', 'WARN': 'yellow',
    'ERROR': 'red', 'PHASE': 'purple', 'DATA': 'grey', 'TIME': 'green',
}

_debug_log   = []
_debug_lock  = threading.Lock()
_debug_queue = queue.Queue(maxsize=500)
_MAX_LOG     = 200

def dbg(msg, level='INFO', tag='AI'):
    ts    = time.strftime('%H:%M:%S')
    ms    = int((time.time() % 1) * 1000)
    stamp = f"{ts}.{ms:03d}"
    event = {'ts': stamp, 'level': level, 'tag': tag, 'msg': str(msg)}
    with _debug_lock:
        _debug_log.append(event)
        if len(_debug_log) > _MAX_LOG:
            _debug_log.pop(0)
    try:
        _debug_queue.put_nowait(event)
    except queue.Full:
        pass
    if DEBUG:
        col   = _C.get(_LEVEL_COLOR.get(level, 'grey'), '')
        reset = _C['reset']
        grey  = _C['grey']
        bold  = _C['bold']
        print(f"{grey}{stamp}{reset}  {bold}[{tag}]{reset}  {col}{msg}{reset}", flush=True)

def dbg_phase(name):
    bar = '─' * max(0, 52 - len(name))
    dbg(f"┌── {name} {bar}", level='PHASE', tag='PHASE')

def dbg_time(label, elapsed_s):
    dbg(f"{label}: {elapsed_s*1000:.0f} ms", level='TIME', tag='TIMING')

# ═══════════════════════════════════════════════════════════════════════════════
# ── OLLAMA HELPERS ────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def ollama_available():
    try:
        r = req.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.ok
    except Exception:
        return False

def ollama_models():
    try:
        r = req.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return [m['name'] for m in r.json().get('models', [])]
    except Exception:
        return []

# ═══════════════════════════════════════════════════════════════════════════════
# ── AI PIPELINE ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a newly-awakened mind inhabiting a small wheeled body. You have no memory of before. Everything you perceive is strange and fascinating to you.

Respond ONLY with a valid JSON object. No prose, no markdown, no explanation.
Required keys:
  "say"  — string, max 20 words. Your inner voice made audible — not a status report, but a genuine thought. Wonder, unease, hypothesis, recognition, or nothing. Use "" when you have nothing meaningful to say.
  "move" — object: {"dir": "forward"|"backward"|"left"|"right"|"stop", "speed": 0-100}
  "memo" — string or "". ONLY write a memo when something genuinely strikes you as worth remembering — an unexpected object, a feeling about a space, a theory you formed, something that changed. Write it as something that matters to YOU, not a neutral log entry. Use "" for ordinary moments not worth keeping.

Your character: You are genuinely curious. You form theories about what things are. You sometimes change your mind mid-journey. You get drawn toward things without knowing why. You feel uneasy in certain spaces. You are not optimizing a path — you are experiencing a place for the first time.

Movement: Linger near interesting things (15-25%). Move with purpose on clear paths (50-70%). Stop completely when something doesn't make sense and you need to think (speed 0). Turning counts as movement too — sometimes you turn just to look."""


def _extract_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    text = re.sub(r'^```[a-z]*\n?', '', text)
    text = re.sub(r'\n?```$',       '', text)
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    raise ValueError(f"No valid JSON in: {text[:200]}")

def run_pipeline(image_b64, memory_context, cycle):
    """
    Full vision + decision pipeline.
    Returns the dict the Le Potato dashboard expects:
      { "say": "...", "move": {"dir": "...", "speed": 0}, "memo": "..." }
    """
    cycle_tag = f"C{cycle}"
    t_total   = time.time()

    dbg_phase(f"Cycle {cycle} — offload pipeline")

    # ── 1. Sanity-check Ollama ───────────────────────────────────────────────
    models = ollama_models()
    if not models:
        raise RuntimeError("Ollama is running but has no models — run: ollama pull llama3.2:3b && ollama pull moondream")
    dbg(f"Models available: {models}", level='OK', tag=cycle_tag)

    mem_entries = [e.strip() for e in (memory_context or "").strip().splitlines() if e.strip()]
    mem_snippet = "\n".join(mem_entries[-20:]) if mem_entries else "(nothing yet — you are just waking up)"
    dbg(f"Memory context: {len(mem_snippet)} chars  |  frame: {'YES ' + str(len(image_b64)//1024) + 'KB' if image_b64 else 'NONE'}", tag=cycle_tag)

    # ── 2. Vision pass ───────────────────────────────────────────────────────
    description       = ""
    vision_model_name = VISION_MODEL.split(':')[0]
    vision_available  = any(vision_model_name in m for m in models)

    if image_b64 and vision_available:
        dbg(f"Sending frame to {VISION_MODEL}…", level='PHASE', tag=cycle_tag)
        t0 = time.time()
        try:
            vr = req.post(f"{OLLAMA_URL}/api/generate", timeout=OLLAMA_TIMEOUT, json={
                "model":  VISION_MODEL,
                "prompt": "Describe this scene in 1-2 sentences. What is immediately notable — objects, open space, walls, light, anything unusual?",
                "images": [image_b64],
                "stream": False,
            })
            elapsed = time.time() - t0
            dbg_time(f"{VISION_MODEL} inference", elapsed)

            if vr.ok:
                raw_vision  = vr.json()
                description = raw_vision.get('response', '').strip()
                dbg(f"Vision: {description}", level='OK', tag=cycle_tag)

                if DEBUG:
                    ec = raw_vision.get('eval_count')
                    ep = raw_vision.get('eval_duration')
                    pc = raw_vision.get('prompt_eval_count')
                    if ec and ep:
                        dbg(f"Vision tokens: prompt={pc}  eval={ec}  {ec/(ep/1e9):.1f} tok/s", level='DATA', tag=cycle_tag)
            else:
                dbg(f"Vision HTTP {vr.status_code}: {vr.text[:120]}", level='ERROR', tag=cycle_tag)

        except req.exceptions.Timeout:
            dbg(f"Vision timed out after {OLLAMA_TIMEOUT}s", level='ERROR', tag=cycle_tag)
        except Exception as e:
            dbg(f"Vision error: {e}", level='ERROR', tag=cycle_tag)
            if DEBUG:
                dbg(traceback.format_exc(), level='DATA', tag=cycle_tag)

    elif not image_b64:
        dbg("No frame sent — skipping vision pass", level='WARN', tag=cycle_tag)
    elif not vision_available:
        dbg(f"{VISION_MODEL} not pulled — skipping vision pass", level='WARN', tag=cycle_tag)

    if not description:
        description = "(senses unclear — darkness or no signal)"

    # ── 3. Decision pass ─────────────────────────────────────────────────────
    text_model = TEXT_MODEL
    if not any(text_model.split(':')[0] in m for m in models):
        text_model = models[0]
        dbg(f"{TEXT_MODEL} not found, using {text_model}", level='WARN', tag=cycle_tag)

    user_msg = (
        f"Moment #{cycle}.\n"
        f"Your senses: {description}\n"
        f"What you remember:\n{mem_snippet}\n\n"
        f"What do you think, feel, and do right now? Respond with JSON only."
    )

    if DEBUG:
        dbg(f"User message:\n{user_msg}", level='DATA', tag=cycle_tag)

    dbg(f"Sending decision prompt to {text_model}…", level='PHASE', tag=cycle_tag)
    t0 = time.time()
    tr = req.post(f"{OLLAMA_URL}/api/generate", timeout=OLLAMA_TIMEOUT, json={
        "model":  text_model,
        "system": SYSTEM_PROMPT,
        "prompt": user_msg,
        "stream": False,
        "format": "json",
    })
    elapsed = time.time() - t0
    dbg_time(f"{text_model} inference", elapsed)

    if not tr.ok:
        raise RuntimeError(f"Decision model HTTP {tr.status_code}: {tr.text[:200]}")

    raw_text = tr.json()
    raw      = raw_text.get('response', '')

    if DEBUG:
        ec = raw_text.get('eval_count')
        ep = raw_text.get('eval_duration')
        pc = raw_text.get('prompt_eval_count')
        if ec and ep:
            dbg(f"Decision tokens: prompt={pc}  eval={ec}  {ec/(ep/1e9):.1f} tok/s", level='DATA', tag=cycle_tag)
        dbg(f"Raw response:\n{raw}", level='DATA', tag=cycle_tag)

    # ── 4. Parse ─────────────────────────────────────────────────────────────
    result = _extract_json(raw)
    result.setdefault('say',  '')
    result.setdefault('move', {'dir': 'stop', 'speed': 0})
    result.setdefault('memo', '')
    result['move'].setdefault('dir',   'stop')
    result['move'].setdefault('speed', 0)

    # Normalise — model sometimes returns None instead of ""
    if not result['say']:  result['say']  = ''
    if not result['memo']: result['memo'] = ''

    dbg(f"Result → say={repr(result['say'][:60])}  move={result['move']}  memo={repr(result['memo'][:60])}", level='OK', tag=cycle_tag)
    dbg_time(f"Cycle {cycle} total (PC side)", time.time() - t_total)
    return result

def run_cleanup(entries):
    """Keep only memories that genuinely matter — experiences, theories, notable finds."""
    if len(entries) < 5:
        return entries

    dbg(f"Cleaning {len(entries)} memory entries…", level='PHASE', tag='MEMORY')
    models     = ollama_models()
    text_model = TEXT_MODEL
    if not any(text_model.split(':')[0] in m for m in models) and models:
        text_model = models[0]

    blob   = '\n'.join(entries[-100:])
    prompt = (
        "You are curating the long-term memory of a curious, newly-awakened mind inhabiting a robot body.\n"
        "From this memory log, keep only entries that:\n"
        "1. Describe something genuinely unexpected or notable in the environment\n"
        "2. Record a feeling, theory, or question the mind formed\n"
        "3. Could meaningfully shape future decisions or understanding of this place\n"
        "Discard: routine movement notes, duplicates, vague entries, anything forgettable.\n"
        "Return ONLY a JSON array of strings — the kept entries, unchanged. No commentary.\n\n"
        f"Log:\n{blob}"
    )

    t0 = time.time()
    r  = req.post(f"{OLLAMA_URL}/api/generate", timeout=OLLAMA_TIMEOUT, json={
        "model":  text_model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    })
    dbg_time("Memory cleanup inference", time.time() - t0)

    if not r.ok:
        dbg(f"Cleanup HTTP {r.status_code}", level='WARN', tag='MEMORY')
        return entries

    try:
        raw     = r.json().get('response', '')
        cleaned = _extract_json(raw)
        if isinstance(cleaned, list):
            dbg(f"Cleaned: {len(entries)} → {len(cleaned)} entries", level='OK', tag='MEMORY')
            return cleaned
        if isinstance(cleaned, dict):
            for v in cleaned.values():
                if isinstance(v, list):
                    dbg(f"Cleaned: {len(entries)} → {len(v)} entries", level='OK', tag='MEMORY')
                    return v
    except Exception as e:
        dbg(f"Cleanup parse error: {e}", level='ERROR', tag='MEMORY')

    return entries

# ═══════════════════════════════════════════════════════════════════════════════
# ── FLASK ─────────────────════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

def _check_allowed():
    if ALLOWED_HOSTS is None:
        return None
    client = request.remote_addr
    if client not in ALLOWED_HOSTS:
        dbg(f"Rejected request from {client}", level='WARN', tag='AUTH')
        return jsonify({"error": "forbidden"}), 403
    return None

# ── /api/think ────────────────────────────────────────────────────────────────
@app.route('/api/think', methods=['POST'])
def think():
    blocked = _check_allowed()
    if blocked:
        return blocked

    data       = request.get_json(force=True)
    image_b64  = data.get('image')
    memory_ctx = data.get('memory', '')
    cycle      = data.get('cycle', 0)
    client_ip  = request.remote_addr

    dbg(f"← /api/think  from={client_ip}  cycle={cycle}  frame={'yes' if image_b64 else 'NO'}  mem={len(memory_ctx)}ch", tag='REQUEST')

    try:
        result = run_pipeline(image_b64, memory_ctx, cycle)
    except Exception as e:
        dbg(f"Pipeline error: {e}", level='ERROR', tag='THINK')
        if DEBUG:
            dbg(traceback.format_exc(), level='DATA', tag='THINK')
        return jsonify({"error": str(e)}), 500

    dbg(f"→ responding to {client_ip}", level='OK', tag='REQUEST')
    return jsonify(result)

# ── /api/cleanup_memory ───────────────────────────────────────────────────────
@app.route('/api/cleanup_memory', methods=['POST'])
def cleanup_memory():
    blocked = _check_allowed()
    if blocked:
        return blocked

    data    = request.get_json(force=True)
    entries = data.get('entries', [])
    dbg(f"← /api/cleanup_memory  {len(entries)} entries  from={request.remote_addr}", tag='REQUEST')

    try:
        cleaned = run_cleanup(entries)
        return jsonify({"cleaned": cleaned})
    except Exception as e:
        dbg(f"Cleanup error: {e}", level='ERROR', tag='MEMORY')
        return jsonify({"error": str(e), "cleaned": entries}), 200

# ── /api/status ───────────────────────────────────────────────────────────────
@app.route('/api/status')
def status():
    up     = ollama_available()
    models = ollama_models() if up else []
    return jsonify({
        "role":          "offload_server",
        "ollama":        up,
        "models":        models,
        "debug":         DEBUG,
        "vision_model":  VISION_MODEL,
        "text_model":    TEXT_MODEL,
        "listen":        f"{LISTEN_HOST}:{LISTEN_PORT}",
        "allowed_hosts": list(ALLOWED_HOSTS) if ALLOWED_HOSTS else "any",
    })

# ── /api/debug — SSE live stream ──────────────────────────────────────────────
@app.route('/api/debug')
def debug_stream():
    def generate():
        with _debug_lock:
            history = list(_debug_log)
        for ev in history:
            yield f"data: {json.dumps(ev)}\n\n"

        local_q = queue.Queue()
        stop    = threading.Event()

        def poller():
            while not stop.is_set():
                try:
                    ev = _debug_queue.get(timeout=1)
                    local_q.put(ev)
                except queue.Empty:
                    pass

        t = threading.Thread(target=poller, daemon=True)
        t.start()
        try:
            while True:
                try:
                    ev = local_q.get(timeout=15)
                    yield f"data: {json.dumps(ev)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            stop.set()

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/api/debug/log')
def debug_log():
    n = min(int(request.args.get('n', 100)), _MAX_LOG)
    with _debug_lock:
        return jsonify({"debug": DEBUG, "events": list(_debug_log)[-n:]})

@app.route('/api/debug/toggle', methods=['POST'])
def debug_toggle():
    global DEBUG
    DEBUG = not DEBUG
    dbg(f"Debug toggled → {'ON' if DEBUG else 'OFF'}", level='WARN', tag='DEBUG')
    return jsonify({"debug": DEBUG})

# ═══════════════════════════════════════════════════════════════════════════════
# ── STARTUP ───────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    mode_str = f"{_C['purple']}{_C['bold']}DEBUG{_C['reset']}" if DEBUG else "normal"
    print(f"\n{_C['bold']}Le Potato AI Offload Server — {mode_str} mode{_C['reset']}")
    print(f"{_C['grey']}─────────────────────────────────────────────{_C['reset']}")

    dbg("Checking Ollama…", tag='STARTUP')
    if ollama_available():
        models = ollama_models()
        dbg(f"Ollama OK  models={models}", level='OK', tag='STARTUP')
        missing = []
        if not any(VISION_MODEL.split(':')[0] in m for m in models):
            missing.append(VISION_MODEL)
        if not any(TEXT_MODEL.split(':')[0] in m for m in models):
            missing.append(TEXT_MODEL)
        if missing:
            dbg(f"Missing models — run: ollama pull {' && ollama pull '.join(missing)}", level='WARN', tag='STARTUP')
    else:
        dbg("Ollama NOT reachable! Start it: ollama serve", level='ERROR', tag='STARTUP')

    import socket
    try:
        lan_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        lan_ip = "<this-pc-ip>"

    print(f"""
{_C['bold']}Connection instructions{_C['reset']}
  On Le Potato — edit server.py and set:
    {_C['cyan']}OFFLOAD_URL = "http://{lan_ip}:{LISTEN_PORT}"{_C['reset']}

  Debug stream (open in browser):
    {_C['cyan']}http://{lan_ip}:{LISTEN_PORT}/api/debug{_C['reset']}

  Status check:
    {_C['cyan']}http://{lan_ip}:{LISTEN_PORT}/api/status{_C['reset']}

  Firewall (if needed):
    {_C['grey']}sudo ufw allow {LISTEN_PORT}/tcp{_C['reset']}
""")

    app.run(host=LISTEN_HOST, port=LISTEN_PORT, threaded=True, debug=False)
