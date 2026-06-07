#!/usr/bin/env python3
"""
Le Potato AI Body — backend server (local Ollama edition)
Ubuntu 22.04 LTS target (AML-S905X-CC / Le Potato)

Install deps:
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull llama3.2:3b          # text + JSON (~2GB, fits in 4GB RAM)
  ollama pull moondream            # vision model (~1.7GB) — sees camera frames
  sudo pip3 install flask lgpio pillow requests

GPIO wiring (Le Potato AML-S905X-CC 40-pin header):
  Servo signal  → physical pin 33  (gpiochip0 line 10)
  DC motor PWM  → physical pin 12  (gpiochip1 line 116)
  DC motor dir1 → physical pin 16  (gpiochip1 line 118)
  DC motor dir2 → physical pin 18  (gpiochip1 line 119)
  GND           → pins 6, 9, 14...

Run:
  openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 3650 -nodes -subj "/CN=lepotato.local"
  sudo python3 server.py
  Open browser → https://<lepotato-ip>:5000  (accept the cert warning)
"""

import json
import os
import re
import time
import threading
import requests as req

from flask import Flask, request, jsonify, send_from_directory

# ── OLLAMA CONFIG ─────────────────────────────────────────────────────────────
OLLAMA_URL        = "http://localhost:11434"
VISION_MODEL      = "moondream"       # handles camera frames
TEXT_MODEL        = "llama3.2:3b"     # handles text-only + memory cleanup
OLLAMA_TIMEOUT    = 60                # seconds — 3b on ARM can be slow

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

# ── GPIO via lgpio ────────────────────────────────────────────────────────────
CHIP_AO   = 0
CHIP_MAIN = 1

LINE_SERVO   = (CHIP_AO,   10)
LINE_DC_PWM  = (CHIP_MAIN, 116)
LINE_DC_DIR1 = (CHIP_MAIN, 118)
LINE_DC_DIR2 = (CHIP_MAIN, 119)

SERVO_MIN_PW = 500
SERVO_MAX_PW = 2500
SERVO_HZ     = 50

try:
    import lgpio
    _h = {}

    def _chip(idx):
        if idx not in _h:
            _h[idx] = lgpio.gpiochip_open(idx)
        return _h[idx]

    lgpio.gpio_claim_output(_chip(LINE_SERVO[0]),   LINE_SERVO[1])
    lgpio.gpio_claim_output(_chip(LINE_DC_PWM[0]),  LINE_DC_PWM[1])
    lgpio.gpio_claim_output(_chip(LINE_DC_DIR1[0]), LINE_DC_DIR1[1])
    lgpio.gpio_claim_output(_chip(LINE_DC_DIR2[0]), LINE_DC_DIR2[1])

    GPIO_AVAILABLE = True
    print("[GPIO] lgpio ready")

except Exception as e:
    print(f"[GPIO] Not available ({e}). Motor commands will be logged only.")
    GPIO_AVAILABLE = False

def _write(line_tuple, value):
    if not GPIO_AVAILABLE:
        return
    chip, line = line_tuple
    lgpio.gpio_write(_chip(chip), line, value)

def _pwm(line_tuple, freq, duty_pct):
    if not GPIO_AVAILABLE:
        return
    chip, line = line_tuple
    lgpio.tx_pwm(_chip(chip), line, freq, max(0.0, min(100.0, duty_pct)))

def setup_gpio():
    if not GPIO_AVAILABLE:
        return
    set_servo(90)
    set_dc_motor(0, 'stop')
    print("[GPIO] Servo centred, DC motor stopped.")

def set_servo(angle_deg):
    if not GPIO_AVAILABLE:
        return
    angle_deg = max(0.0, min(180.0, float(angle_deg)))
    pw_us = SERVO_MIN_PW + (angle_deg / 180.0) * (SERVO_MAX_PW - SERVO_MIN_PW)
    duty  = (pw_us / 20_000.0) * 100.0
    _pwm(LINE_SERVO, SERVO_HZ, duty)

def set_dc_motor(speed_pct, direction):
    if not GPIO_AVAILABLE:
        return
    speed_pct = max(0.0, min(100.0, float(speed_pct)))
    if direction == 'stop' or speed_pct == 0:
        _pwm(LINE_DC_PWM, 1000, 0)
        _write(LINE_DC_DIR1, 0)
        _write(LINE_DC_DIR2, 0)
    elif direction == 'forward':
        _write(LINE_DC_DIR1, 1)
        _write(LINE_DC_DIR2, 0)
        _pwm(LINE_DC_PWM, 1000, speed_pct)
    elif direction == 'backward':
        _write(LINE_DC_DIR1, 0)
        _write(LINE_DC_DIR2, 1)
        _pwm(LINE_DC_PWM, 1000, speed_pct)

def apply_move_command(move):
    raw_dir = move.get('dir', 'stop').lower()
    speed   = float(move.get('speed', 0))
    angle_map = {'forward': 90, 'backward': 90, 'left': 45, 'right': 135, 'stop': 90}
    servo_angle = angle_map.get(raw_dir, 90)
    dc_dir = raw_dir if raw_dir in ('forward', 'backward', 'stop') else 'forward'
    set_servo(servo_angle)
    set_dc_motor(speed, dc_dir)
    print(f"[MOTOR] dir={raw_dir} speed={speed:.0f}% servo={servo_angle}°")

# ── MEMORY ────────────────────────────────────────────────────────────────────
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

# ── AI — OLLAMA ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the mind of a small wheeled robot with a camera.
Respond ONLY with a valid JSON object. No prose, no markdown, no explanation.
Required keys:
  "say"  — string, max 20 words to speak aloud. Empty string if silent.
  "move" — object: {"dir": "forward"|"backward"|"left"|"right"|"stop", "speed": 0-100}
  "memo" — string, one short fact about what you observed. Empty string if nothing notable.
Rules: avoid obstacles, be curious, move slowly (20-40%) in new places, fast (60%+) only on clear paths."""

def _extract_json(text):
    """Pull the first {...} block out of a model response robustly."""
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Strip markdown fences
    text = re.sub(r'^```[a-z]*\n?', '', text)
    text = re.sub(r'\n?```$', '', text)
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # Find first {...} block
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    raise ValueError(f"No valid JSON found in response: {text[:200]}")

def _fallback_response():
    return {"say": "", "move": {"dir": "stop", "speed": 0}, "memo": ""}

def query_local(image_b64, memory_context, cycle):
    models = ollama_models()
    mem_snippet = (memory_context or "").strip()[-1500:] or "(none)"

    # ── Vision pass (moondream describes what it sees) ──────────────────────
    description = ""
    if image_b64 and VISION_MODEL.split(':')[0] in ' '.join(models):
        try:
            vr = req.post(f"{OLLAMA_URL}/api/generate", timeout=OLLAMA_TIMEOUT, json={
                "model": VISION_MODEL,
                "prompt": "Describe this scene briefly in 1-2 sentences. Focus on obstacles, open paths, and objects.",
                "images": [image_b64],
                "stream": False,
            })
            if vr.ok:
                description = vr.json().get('response', '').strip()
                print(f"[VISION] {description}")
        except Exception as e:
            print(f"[VISION] Error: {e}")

    if not description:
        description = "(no camera image available)"

    # ── Decision pass (llama3.2:3b decides what to do) ───────────────────────
    user_msg = (
        f"Cycle #{cycle}.\n"
        f"What the camera sees: {description}\n"
        f"Recent memory:\n{mem_snippet}\n\n"
        f"Respond with JSON only."
    )

    text_model = TEXT_MODEL
    # Fallback: use whatever model is installed if preferred isn't available
    if text_model.split(':')[0] not in ' '.join(models) and models:
        text_model = models[0]
        print(f"[AI] {TEXT_MODEL} not found, using {text_model}")

    tr = req.post(f"{OLLAMA_URL}/api/generate", timeout=OLLAMA_TIMEOUT, json={
        "model": text_model,
        "system": SYSTEM_PROMPT,
        "prompt": user_msg,
        "stream": False,
        "format": "json",   # Ollama JSON mode — forces valid JSON output
    })

    if not tr.ok:
        raise RuntimeError(f"Ollama HTTP {tr.status_code}: {tr.text[:200]}")

    raw = tr.json().get('response', '')
    result = _extract_json(raw)

    # Ensure required keys exist
    result.setdefault('say', '')
    result.setdefault('move', {'dir': 'stop', 'speed': 0})
    result.setdefault('memo', '')
    result['move'].setdefault('dir', 'stop')
    result['move'].setdefault('speed', 0)

    return result

def cleanup_memory_local(entries):
    """Use the text model to deduplicate and clean the memory log."""
    if len(entries) < 5:
        return entries

    models = ollama_models()
    text_model = TEXT_MODEL
    if text_model.split(':')[0] not in ' '.join(models) and models:
        text_model = models[0]

    blob = '\n'.join(entries[-100:])
    prompt = (
        "You are a memory curator for a robot.\n"
        "Clean this memory log:\n"
        "1. Remove duplicate or near-duplicate entries.\n"
        "2. Remove contradicted facts (keep the newer one).\n"
        "3. Remove vague entries.\n"
        "Return ONLY a JSON array of strings. No commentary.\n\n"
        f"Log:\n{blob}"
    )

    r = req.post(f"{OLLAMA_URL}/api/generate", timeout=OLLAMA_TIMEOUT, json={
        "model": text_model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    })

    if not r.ok:
        return entries

    try:
        raw = r.json().get('response', '')
        cleaned = _extract_json(raw)
        if isinstance(cleaned, list):
            return cleaned
        # Model may return {"entries": [...]}
        if isinstance(cleaned, dict):
            for v in cleaned.values():
                if isinstance(v, list):
                    return v
    except Exception as e:
        print(f"[CLEANUP] Parse error: {e}")

    return entries

# ── FLASK APP ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='.')

@app.route('/')
def index():
    return send_from_directory('.', 'lepotato_dashboard.html')

@app.route('/api/status')
def status():
    up = ollama_available()
    models = ollama_models() if up else []
    return jsonify({
        "ollama": up,
        "models": models,
        "gpio": GPIO_AVAILABLE,
        "vision_model": VISION_MODEL,
        "text_model": TEXT_MODEL,
    })

@app.route('/api/think', methods=['POST'])
def think():
    data       = request.get_json(force=True)
    image_b64  = data.get('image')
    memory_ctx = data.get('memory', '')
    cycle      = data.get('cycle', 0)

    try:
        result = query_local(image_b64, memory_ctx, cycle)
    except Exception as e:
        print(f"[THINK] Error: {e}")
        return jsonify({"error": str(e)}), 500

    if 'move' in result:
        threading.Thread(target=apply_move_command, args=(result['move'],), daemon=True).start()

    if result.get('memo'):
        append_memory(result['memo'])

    return jsonify(result)

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

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    setup_gpio()

    print(f"[AI] Checking Ollama... ", end='', flush=True)
    if ollama_available():
        models = ollama_models()
        print(f"OK — models: {models}")
        if not models:
            print("[AI] WARNING: No models pulled yet.")
            print(f"[AI] Run: ollama pull {TEXT_MODEL}")
            print(f"[AI] Run: ollama pull {VISION_MODEL}")
    else:
        print("NOT RUNNING")
        print("[AI] Start Ollama with: ollama serve")
        print(f"[AI] Then pull models:  ollama pull {TEXT_MODEL} && ollama pull {VISION_MODEL}")

    cert = ('cert.pem', 'key.pem')
    if os.path.exists('cert.pem') and os.path.exists('key.pem'):
        print("[SERVER] HTTPS enabled — open https://<ip>:5000")
        ssl_ctx = cert
    else:
        print("[SERVER] No cert found — camera will not work.")
        print("[SERVER] Fix: openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 3650 -nodes -subj '/CN=lepotato.local'")
        ssl_ctx = None

    app.run(host='0.0.0.0', port=5000, threaded=True, ssl_context=ssl_ctx)
