#!/usr/bin/env python3
"""
Le Potato AI Body — backend server (local Ollama edition)
Ubuntu 22.04 LTS target (AML-S905X-CC / Le Potato)

Install deps:
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull llama3.2:3b
  ollama pull moondream
  sudo pip3 install flask lgpio pillow requests --break-system-packages

GPIO wiring — verified chip/line via `lgpio info PIN gpiod`:
  Physical pin 12  → chip 0  line 6   — DC motor PWM  (lgpio tx_pwm works here)
  Physical pin 16  → chip 1  line 93  — DC motor dir1
  Physical pin 18  → chip 1  line 94  — DC motor dir2
  Physical pin 33  → chip 1  line 85  — Servo signal  (bit-bang, NOT tx_pwm)
  GND              → pins 6, 9, 14…

NOTE: lgpio.tx_pwm() only works on gpiochip0. Pins 16/18/33 are on gpiochip1
so direction pins are plain digital writes and servo uses a background thread.

Run (normal):
  sudo python3 server.py

Run (debug — full AI pipeline logging + /api/debug SSE stream):
  sudo python3 server.py --debug
  Or set env:  export LEPOTATO_DEBUG=1

Debug endpoints:
  GET  /api/debug        — SSE stream of live debug events (open in browser)
  GET  /api/debug/log    — last 200 debug lines as JSON
  POST /api/debug/toggle — toggle debug on/off at runtime

TLS (required for webcam from a remote browser):
  openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 3650 -nodes -subj "/CN=lepotato.local"
  sudo python3 server.py [--debug]
  Open → https://<lepotato-ip>:5000
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

from flask import Flask, request, jsonify, send_from_directory, Response

# ═══════════════════════════════════════════════════════════════════════════════
# ── DEBUG SYSTEM ──────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

DEBUG = '--debug' in sys.argv or os.environ.get('LEPOTATO_DEBUG', '0') == '1'

# ANSI colours for terminal output
_C = {
    'reset':  '\033[0m',
    'grey':   '\033[90m',
    'cyan':   '\033[96m',
    'green':  '\033[92m',
    'yellow': '\033[93m',
    'red':    '\033[91m',
    'purple': '\033[95m',
    'bold':   '\033[1m',
}

# Ring buffer of recent debug events for /api/debug/log
_debug_log   = []          # list of dicts
_debug_lock  = threading.Lock()
_debug_queue = queue.Queue()   # fed to SSE subscribers
_MAX_LOG     = 200

_LEVEL_COLOR = {
    'INFO':  'cyan',
    'OK':    'green',
    'WARN':  'yellow',
    'ERROR': 'red',
    'PHASE': 'purple',
    'DATA':  'grey',
    'TIME':  'green',
}

def dbg(msg, level='INFO', tag='DEBUG'):
    """
    Emit a debug event.  Always goes to the ring buffer and SSE queue.
    Only prints to terminal if DEBUG is True.
    level: INFO | OK | WARN | ERROR | PHASE | DATA | TIME
    """
    ts    = time.strftime('%H:%M:%S')
    ms    = int((time.time() % 1) * 1000)
    stamp = f"{ts}.{ms:03d}"

    event = {
        'ts':    stamp,
        'level': level,
        'tag':   tag,
        'msg':   str(msg),
    }

    with _debug_lock:
        _debug_log.append(event)
        if len(_debug_log) > _MAX_LOG:
            _debug_log.pop(0)

    # Always push to SSE queue (subscribers decide)
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
    """Print a prominent phase banner in the terminal."""
    bar = '─' * (52 - len(name))
    dbg(f"┌── {name} {bar}", level='PHASE', tag='PHASE')

def dbg_time(label, elapsed_s):
    dbg(f"{label}: {elapsed_s*1000:.0f} ms", level='TIME', tag='TIMING')


# ═══════════════════════════════════════════════════════════════════════════════
# ── OLLAMA CONFIG ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

OLLAMA_URL     = "http://localhost:11434"
VISION_MODEL   = "moondream"
TEXT_MODEL     = "llama3.2:3b"
OLLAMA_TIMEOUT = 60

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
# ── GPIO PIN MAP ──────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
# Verified with: lgpio info <phys_pin> gpiod
# Format: (chip_index, line_number)

PIN_DC_PWM  = (0,  6)   # phys 12 — chip 0 line 6  — supports tx_pwm
PIN_DC_DIR1 = (1, 93)   # phys 16 — chip 1 line 93
PIN_DC_DIR2 = (1, 94)   # phys 18 — chip 1 line 94
PIN_SERVO   = (1, 85)   # phys 33 — chip 1 line 85 — bit-banged PWM

SERVO_HZ   = 50
DC_PWM_HZ  = 1000

GPIO_AVAILABLE = False
_handles     = {}
_servo_angle = 90
_servo_lock  = threading.Lock()
_servo_thread = None
_servo_stop   = threading.Event()

try:
    import lgpio

    def _chip(idx):
        if idx not in _handles:
            _handles[idx] = lgpio.gpiochip_open(idx)
        return _handles[idx]

    lgpio.gpio_claim_output(_chip(PIN_DC_PWM[0]),  PIN_DC_PWM[1])
    lgpio.gpio_claim_output(_chip(PIN_DC_DIR1[0]), PIN_DC_DIR1[1])
    lgpio.gpio_claim_output(_chip(PIN_DC_DIR2[0]), PIN_DC_DIR2[1])
    lgpio.gpio_claim_output(_chip(PIN_SERVO[0]),   PIN_SERVO[1])

    GPIO_AVAILABLE = True
    dbg("lgpio ready — chip0 line6 (DC PWM), chip1 lines 85/93/94", level='OK', tag='GPIO')

except Exception as e:
    dbg(f"Not available: {e}  — motor commands logged only", level='WARN', tag='GPIO')

# ── SERVO BIT-BANG ────────────────────────────────────────────────────────────

SERVO_PERIOD = 1.0 / SERVO_HZ
SERVO_MIN_PW = 0.0005
SERVO_MAX_PW = 0.0025

def _servo_pw(angle_deg):
    angle_deg = max(0.0, min(180.0, float(angle_deg)))
    return SERVO_MIN_PW + (angle_deg / 180.0) * (SERVO_MAX_PW - SERVO_MIN_PW)

def _servo_loop():
    chip, line = PIN_SERVO
    h = _chip(chip)
    while not _servo_stop.is_set():
        with _servo_lock:
            angle = _servo_angle
        pw  = _servo_pw(angle)
        low = SERVO_PERIOD - pw
        lgpio.gpio_write(h, line, 1)
        time.sleep(pw)
        lgpio.gpio_write(h, line, 0)
        time.sleep(low)

def start_servo_thread():
    global _servo_thread
    if not GPIO_AVAILABLE:
        return
    _servo_stop.clear()
    _servo_thread = threading.Thread(target=_servo_loop, daemon=True)
    _servo_thread.start()
    dbg("Bit-bang thread started", level='OK', tag='SERVO')

def set_servo(angle_deg):
    global _servo_angle
    if not GPIO_AVAILABLE:
        return
    with _servo_lock:
        _servo_angle = max(0.0, min(180.0, float(angle_deg)))

# ── DC MOTOR ──────────────────────────────────────────────────────────────────

def _write(pin_tuple, value):
    if not GPIO_AVAILABLE:
        return
    chip, line = pin_tuple
    lgpio.gpio_write(_chip(chip), line, value)

def _pwm(pin_tuple, freq, duty_pct):
    if not GPIO_AVAILABLE:
        return
    chip, line = pin_tuple
    lgpio.tx_pwm(_chip(chip), line, freq, max(0.0, min(100.0, duty_pct)))

def set_dc_motor(speed_pct, direction):
    if not GPIO_AVAILABLE:
        return
    speed_pct = max(0.0, min(100.0, float(speed_pct)))
    if direction == 'stop' or speed_pct == 0:
        _pwm(PIN_DC_PWM, DC_PWM_HZ, 0)
        _write(PIN_DC_DIR1, 0)
        _write(PIN_DC_DIR2, 0)
    elif direction == 'forward':
        _write(PIN_DC_DIR1, 1)
        _write(PIN_DC_DIR2, 0)
        _pwm(PIN_DC_PWM, DC_PWM_HZ, speed_pct)
    elif direction == 'backward':
        _write(PIN_DC_DIR1, 0)
        _write(PIN_DC_DIR2, 1)
        _pwm(PIN_DC_PWM, DC_PWM_HZ, speed_pct)

def setup_gpio():
    if not GPIO_AVAILABLE:
        return
    set_servo(90)
    set_dc_motor(0, 'stop')
    dbg("Servo centred, DC motor stopped", level='OK', tag='GPIO')

def apply_move_command(move):
    raw_dir = move.get('dir', 'stop').lower()
    speed   = float(move.get('speed', 0))
    angle_map = {'forward': 90, 'backward': 90, 'left': 45, 'right': 135, 'stop': 90}
    servo_angle = angle_map.get(raw_dir, 90)
    dc_dir = raw_dir if raw_dir in ('forward', 'backward') else 'stop'
    if raw_dir in ('left', 'right'):
        dc_dir = 'forward'
    set_servo(servo_angle)
    set_dc_motor(speed, dc_dir)
    dbg(f"dir={raw_dir}  speed={speed:.0f}%  servo={servo_angle}°", level='OK', tag='MOTOR')

# ═══════════════════════════════════════════════════════════════════════════════
# ── MEMORY ────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

MEMORY_PATH = os.path.join(os.path.dirname(__file__), 'memory.log')

def load_memory():
    if not os.path.exists(MEMORY_PATH):
        return []
    with open(MEMORY_PATH, 'r') as f:
        return [l.strip() for l in f if l.strip()]

def save_memory(entries):
    with open(MEMORY_PATH, 'w') as f:
        f.write('\n'.join(entries) + '\n')

def append_memory(note):
    ts    = time.strftime('%Y-%m-%d %H:%M:%S')
    entry = f"[{ts}] {note}"
    with open(MEMORY_PATH, 'a') as f:
        f.write(entry + '\n')
    return entry

# ═══════════════════════════════════════════════════════════════════════════════
# ── AI — OLLAMA ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are the mind of a small wheeled robot with a camera.
Respond ONLY with a valid JSON object. No prose, no markdown, no explanation.
Required keys:
  "say"  — string, max 20 words to speak aloud. Empty string if silent.
  "move" — object: {"dir": "forward"|"backward"|"left"|"right"|"stop", "speed": 0-100}
  "memo" — string, one short fact about what you observed. Empty string if nothing notable.
Rules: avoid obstacles, be curious, move slowly (20-40%) in new places, fast (60%+) only on clear paths."""

def _extract_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    text = re.sub(r'^```[a-z]*\n?', '', text)
    text = re.sub(r'\n?```$', '', text)
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
    raise ValueError(f"No valid JSON found in response: {text[:200]}")

def query_local(image_b64, memory_context, cycle):
    cycle_tag = f"C{cycle}"
    t_total   = time.time()

    # ── 1. Check Ollama is up ────────────────────────────────────────────────
    dbg_phase(f"Cycle {cycle} — AI pipeline")
    dbg("Checking Ollama availability…", tag=cycle_tag)
    models = ollama_models()
    if not models:
        dbg("Ollama returned no models — is it running?", level='ERROR', tag=cycle_tag)
        raise RuntimeError("Ollama has no models loaded")
    dbg(f"Available models: {models}", level='OK', tag=cycle_tag)

    mem_snippet = (memory_context or "").strip()[-1500:] or "(none)"
    dbg(f"Memory context: {len(mem_snippet)} chars", tag=cycle_tag)
    if DEBUG:
        dbg(f"Memory snippet:\n{mem_snippet[:300]}{'…' if len(mem_snippet)>300 else ''}", level='DATA', tag=cycle_tag)

    # ── 2. Vision pass ───────────────────────────────────────────────────────
    description = ""
    vision_model_name = VISION_MODEL.split(':')[0]
    vision_available  = any(vision_model_name in m for m in models)

    if image_b64:
        dbg(f"Frame received ({len(image_b64)//1024} KB b64)", tag=cycle_tag)
    else:
        dbg("No frame received from dashboard", level='WARN', tag=cycle_tag)

    if image_b64 and vision_available:
        dbg(f"Sending frame to {VISION_MODEL}…", level='PHASE', tag=cycle_tag)
        t0 = time.time()
        try:
            vr = req.post(f"{OLLAMA_URL}/api/generate", timeout=OLLAMA_TIMEOUT, json={
                "model": VISION_MODEL,
                "prompt": "Describe this scene briefly in 1-2 sentences. Focus on obstacles, open paths, and objects.",
                "images": [image_b64],
                "stream": False,
            })
            elapsed = time.time() - t0
            dbg_time(f"{VISION_MODEL} inference", elapsed)
            if vr.ok:
                raw_vision = vr.json()
                description = raw_vision.get('response', '').strip()
                dbg(f"Vision description: {description}", level='OK', tag=cycle_tag)
                if DEBUG:
                    # Log token counts if Ollama returns them
                    ec = raw_vision.get('eval_count')
                    ep = raw_vision.get('eval_duration')
                    pc = raw_vision.get('prompt_eval_count')
                    if ec and ep:
                        tok_s = ec / (ep / 1e9)
                        dbg(f"Vision tokens: prompt={pc}  eval={ec}  speed={tok_s:.1f} tok/s", level='DATA', tag=cycle_tag)
            else:
                dbg(f"Vision HTTP {vr.status_code}: {vr.text[:120]}", level='ERROR', tag=cycle_tag)
        except req.exceptions.Timeout:
            dbg(f"Vision timed out after {OLLAMA_TIMEOUT}s", level='ERROR', tag=cycle_tag)
        except Exception as e:
            dbg(f"Vision exception: {e}", level='ERROR', tag=cycle_tag)
            if DEBUG:
                dbg(traceback.format_exc(), level='DATA', tag=cycle_tag)
    elif not vision_available:
        dbg(f"{VISION_MODEL} not in model list — skipping vision pass", level='WARN', tag=cycle_tag)
    
    if not description:
        description = "(no camera image available)"
        dbg("Using fallback description", level='WARN', tag=cycle_tag)

    # ── 3. Decision pass ─────────────────────────────────────────────────────
    text_model = TEXT_MODEL
    if not any(text_model.split(':')[0] in m for m in models):
        text_model = models[0]
        dbg(f"{TEXT_MODEL} not found, falling back to {text_model}", level='WARN', tag=cycle_tag)

    user_msg = (
        f"Cycle #{cycle}.\n"
        f"What the camera sees: {description}\n"
        f"Recent memory:\n{mem_snippet}\n\n"
        f"Respond with JSON only."
    )

    if DEBUG:
        dbg(f"System prompt ({len(SYSTEM_PROMPT)} chars):\n{SYSTEM_PROMPT}", level='DATA', tag=cycle_tag)
        dbg(f"User message:\n{user_msg}", level='DATA', tag=cycle_tag)

    dbg(f"Sending decision prompt to {text_model}…", level='PHASE', tag=cycle_tag)
    t0 = time.time()
    tr = req.post(f"{OLLAMA_URL}/api/generate", timeout=OLLAMA_TIMEOUT, json={
        "model": text_model,
        "system": SYSTEM_PROMPT,
        "prompt": user_msg,
        "stream": False,
        "format": "json",
    })
    elapsed = time.time() - t0
    dbg_time(f"{text_model} inference", elapsed)

    if not tr.ok:
        dbg(f"Decision HTTP {tr.status_code}: {tr.text[:200]}", level='ERROR', tag=cycle_tag)
        raise RuntimeError(f"Ollama HTTP {tr.status_code}: {tr.text[:200]}")

    raw_text = tr.json()
    raw      = raw_text.get('response', '')

    if DEBUG:
        ec = raw_text.get('eval_count')
        ep = raw_text.get('eval_duration')
        pc = raw_text.get('prompt_eval_count')
        if ec and ep:
            tok_s = ec / (ep / 1e9)
            dbg(f"Decision tokens: prompt={pc}  eval={ec}  speed={tok_s:.1f} tok/s", level='DATA', tag=cycle_tag)
        dbg(f"Raw model response:\n{raw}", level='DATA', tag=cycle_tag)

    # ── 4. Parse JSON ────────────────────────────────────────────────────────
    dbg("Parsing JSON response…", tag=cycle_tag)
    try:
        result = _extract_json(raw)
    except ValueError as e:
        dbg(f"JSON parse failed: {e}", level='ERROR', tag=cycle_tag)
        raise

    result.setdefault('say', '')
    result.setdefault('move', {'dir': 'stop', 'speed': 0})
    result.setdefault('memo', '')
    result['move'].setdefault('dir', 'stop')
    result['move'].setdefault('speed', 0)

    dbg(f"Parsed result: say={repr(result['say'][:40])}  move={result['move']}  memo={repr(result['memo'][:40])}", level='OK', tag=cycle_tag)
    dbg_time(f"Cycle {cycle} total", time.time() - t_total)

    return result

def cleanup_memory_local(entries):
    if len(entries) < 5:
        return entries

    dbg(f"Cleaning {len(entries)} memory entries…", level='PHASE', tag='MEMORY')
    models = ollama_models()
    text_model = TEXT_MODEL
    if text_model.split(':')[0] not in ' '.join(models) and models:
        text_model = models[0]

    blob   = '\n'.join(entries[-100:])
    prompt = (
        "You are a memory curator for a robot.\n"
        "Clean this memory log:\n"
        "1. Remove duplicate or near-duplicate entries.\n"
        "2. Remove contradicted facts (keep the newer one).\n"
        "3. Remove vague entries.\n"
        "Return ONLY a JSON array of strings. No commentary.\n\n"
        f"Log:\n{blob}"
    )

    t0 = time.time()
    r  = req.post(f"{OLLAMA_URL}/api/generate", timeout=OLLAMA_TIMEOUT, json={
        "model": text_model,
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
            dbg(f"Cleaned {len(entries)} → {len(cleaned)} entries", level='OK', tag='MEMORY')
            return cleaned
        if isinstance(cleaned, dict):
            for v in cleaned.values():
                if isinstance(v, list):
                    dbg(f"Cleaned {len(entries)} → {len(v)} entries", level='OK', tag='MEMORY')
                    return v
    except Exception as e:
        dbg(f"Cleanup parse error: {e}", level='ERROR', tag='MEMORY')

    return entries

# ═══════════════════════════════════════════════════════════════════════════════
# ── FLASK APP ─────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__, static_folder='.')

@app.route('/')
def index():
    return send_from_directory('.', 'lepotato_dashboard.html')

# ── /api/status ───────────────────────────────────────────────────────────────
@app.route('/api/status')
def status():
    up     = ollama_available()
    models = ollama_models() if up else []
    return jsonify({
        "ollama": up,
        "models": models,
        "gpio":   GPIO_AVAILABLE,
        "debug":  DEBUG,
        "vision_model": VISION_MODEL,
        "text_model":   TEXT_MODEL,
        "pin_map": {
            "servo":   f"chip{PIN_SERVO[0]} line{PIN_SERVO[1]} (bit-bang, phys33)",
            "dc_pwm":  f"chip{PIN_DC_PWM[0]} line{PIN_DC_PWM[1]} (tx_pwm, phys12)",
            "dc_dir1": f"chip{PIN_DC_DIR1[0]} line{PIN_DC_DIR1[1]} (phys16)",
            "dc_dir2": f"chip{PIN_DC_DIR2[0]} line{PIN_DC_DIR2[1]} (phys18)",
        }
    })

# ── /api/think ────────────────────────────────────────────────────────────────
@app.route('/api/think', methods=['POST'])
def think():
    data       = request.get_json(force=True)
    image_b64  = data.get('image')
    memory_ctx = data.get('memory', '')
    cycle      = data.get('cycle', 0)
    dbg(f"--- /api/think  cycle={cycle}  image={'yes' if image_b64 else 'NO'}  mem_chars={len(memory_ctx)}", tag='REQUEST')

    try:
        result = query_local(image_b64, memory_ctx, cycle)
    except Exception as e:
        dbg(f"query_local raised: {e}", level='ERROR', tag='THINK')
        if DEBUG:
            dbg(traceback.format_exc(), level='DATA', tag='THINK')
        return jsonify({"error": str(e)}), 500

    if 'move' in result:
        threading.Thread(target=apply_move_command, args=(result['move'],), daemon=True).start()

    if result.get('memo'):
        append_memory(result['memo'])

    return jsonify(result)

# ── /api/cleanup_memory ───────────────────────────────────────────────────────
@app.route('/api/cleanup_memory', methods=['POST'])
def cleanup_memory():
    data    = request.get_json(force=True)
    entries = data.get('entries', [])
    try:
        cleaned = cleanup_memory_local(entries)
        save_memory(cleaned)
        return jsonify({"cleaned": cleaned})
    except Exception as e:
        return jsonify({"error": str(e), "cleaned": entries}), 200

# ── /api/debug — SSE live stream ──────────────────────────────────────────────
@app.route('/api/debug')
def debug_stream():
    """
    Server-Sent Events stream of real-time debug events.
    Open in browser:  https://<ip>:5000/api/debug
    Or curl:          curl -k https://<ip>:5000/api/debug
    """
    def generate():
        # First flush the existing log so the client has context
        with _debug_lock:
            history = list(_debug_log)
        for ev in history:
            yield f"data: {json.dumps(ev)}\n\n"

        # Then stream live events
        local_q = queue.Queue()
        # Fan out from the global queue by polling (simple, avoids subscriber list)
        # We use a per-request queue fed by a poller thread
        stop = threading.Event()

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
                    # keepalive ping
                    yield ": keepalive\n\n"
        except GeneratorExit:
            stop.set()

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

# ── /api/debug/log — last N events as JSON ────────────────────────────────────
@app.route('/api/debug/log')
def debug_log():
    n = min(int(request.args.get('n', 100)), _MAX_LOG)
    with _debug_lock:
        return jsonify({"debug": DEBUG, "events": list(_debug_log)[-n:]})

# ── /api/debug/toggle — flip debug at runtime ─────────────────────────────────
@app.route('/api/debug/toggle', methods=['POST'])
def debug_toggle():
    global DEBUG
    DEBUG = not DEBUG
    dbg(f"Debug mode toggled → {'ON' if DEBUG else 'OFF'}", level='WARN', tag='DEBUG')
    return jsonify({"debug": DEBUG})

# ═══════════════════════════════════════════════════════════════════════════════
# ── MAIN ──────────────────────────────────════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    mode_str = f"{_C['purple']}{_C['bold']}DEBUG{_C['reset']}" if DEBUG else "normal"
    print(f"\n{_C['bold']}Le Potato AI Body — starting in {mode_str} mode{_C['reset']}\n")

    if DEBUG:
        print(f"  {_C['cyan']}Live debug stream →  https://<ip>:5000/api/debug{_C['reset']}")
        print(f"  {_C['cyan']}Last 100 events   →  https://<ip>:5000/api/debug/log{_C['reset']}")
        print(f"  {_C['cyan']}Toggle debug      →  POST https://<ip>:5000/api/debug/toggle{_C['reset']}\n")

    setup_gpio()
    start_servo_thread()

    dbg(f"Checking Ollama at {OLLAMA_URL}…", tag='STARTUP')
    if ollama_available():
        models = ollama_models()
        dbg(f"Ollama OK — models: {models}", level='OK', tag='STARTUP')
        if not models:
            dbg(f"No models found! Run: ollama pull {TEXT_MODEL} && ollama pull {VISION_MODEL}", level='ERROR', tag='STARTUP')
    else:
        dbg("Ollama NOT reachable — start with: ollama serve", level='ERROR', tag='STARTUP')
        dbg(f"Pull models after: ollama pull {TEXT_MODEL} && ollama pull {VISION_MODEL}", level='WARN', tag='STARTUP')

    if os.path.exists('cert.pem') and os.path.exists('key.pem'):
        dbg("HTTPS enabled — cert.pem / key.pem found", level='OK', tag='SERVER')
        ssl_ctx = ('cert.pem', 'key.pem')
    else:
        dbg("No TLS cert — webcam from remote browser won't work", level='WARN', tag='SERVER')
        dbg("Fix: openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 3650 -nodes -subj '/CN=lepotato.local'", level='WARN', tag='SERVER')
        ssl_ctx = None

    app.run(host='0.0.0.0', port=5000, threaded=True, ssl_context=ssl_ctx)
