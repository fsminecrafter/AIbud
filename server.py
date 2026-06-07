#!/usr/bin/env python3
"""
Le Potato AI Body — backend server
Ubuntu 22.04 LTS target (AML-S905X-CC / Le Potato)

Install deps:
  pip3 install flask anthropic pigpio pillow --break-system-packages

GPIO wiring (Le Potato AML-S905X-CC 40-pin header):
  Servo signal  → pin 33 (GPIO AO_10)
  DC motor PWM  → pin 12 (GPIO_18)
  DC motor dir1 → pin 16 (GPIO_23)
  DC motor dir2 → pin 18 (GPIO_24)
  GND           → pins 6, 9, 14...

Run:
  python3 server.py
  Open browser → http://<lepotato-ip>:5000
"""

import base64
import json
import os
import time
import threading
from io import BytesIO

from flask import Flask, request, jsonify, send_from_directory
import anthropic

# ── GPIO (comment out if testing without hardware) ───────────────────────────
try:
    import pigpio
    pi = pigpio.pi()
    if not pi.connected:
        raise RuntimeError("pigpio daemon not running — start with: sudo pigpiod")
    GPIO_AVAILABLE = True
    print("[GPIO] pigpio connected")
except Exception as e:
    print(f"[GPIO] Not available ({e}). Motor commands will be logged only.")
    pi = None
    GPIO_AVAILABLE = False

# Pin assignments (BCM numbering via pigpio)
PIN_SERVO     = 33   # Servo PWM signal
PIN_DC_PWM    = 12   # DC motor speed (PWM)
PIN_DC_DIR1   = 16   # DC motor direction A
PIN_DC_DIR2   = 18   # DC motor direction B

SERVO_MIN_PW  = 500   # microseconds — hard left (~0°)
SERVO_MAX_PW  = 2500  # microseconds — hard right (~180°)
SERVO_CENTER  = 1500  # microseconds — center (90°)

def setup_gpio():
    if not GPIO_AVAILABLE:
        return
    pi.set_mode(PIN_SERVO, pigpio.OUTPUT)
    pi.set_mode(PIN_DC_PWM, pigpio.OUTPUT)
    pi.set_mode(PIN_DC_DIR1, pigpio.OUTPUT)
    pi.set_mode(PIN_DC_DIR2, pigpio.OUTPUT)
    pi.set_servo_pulsewidth(PIN_SERVO, SERVO_CENTER)

def set_servo(angle_deg: float):
    """angle_deg: 0–180. 90 = center / straight ahead."""
    if not GPIO_AVAILABLE:
        return
    angle_deg = max(0, min(180, angle_deg))
    pw = int(SERVO_MIN_PW + (angle_deg / 180.0) * (SERVO_MAX_PW - SERVO_MIN_PW))
    pi.set_servo_pulsewidth(PIN_SERVO, pw)

def set_dc_motor(speed_pct: float, direction: str):
    """speed_pct: 0–100. direction: 'forward' | 'backward' | 'stop'."""
    if not GPIO_AVAILABLE:
        return
    speed_pct = max(0, min(100, speed_pct))
    duty = int(speed_pct / 100.0 * 255)

    if direction == 'stop' or speed_pct == 0:
        pi.set_PWM_dutycycle(PIN_DC_PWM, 0)
        pi.write(PIN_DC_DIR1, 0)
        pi.write(PIN_DC_DIR2, 0)
    elif direction == 'forward':
        pi.write(PIN_DC_DIR1, 1)
        pi.write(PIN_DC_DIR2, 0)
        pi.set_PWM_dutycycle(PIN_DC_PWM, duty)
    elif direction == 'backward':
        pi.write(PIN_DC_DIR1, 0)
        pi.write(PIN_DC_DIR2, 1)
        pi.set_PWM_dutycycle(PIN_DC_PWM, duty)

def apply_move_command(move: dict):
    """Apply a move dict from the AI JSON response to the hardware."""
    raw_dir   = move.get('dir', 'stop').lower()
    speed     = float(move.get('speed', 0))

    # Map direction to servo angle + DC motor
    angle_map = {
        'forward':  90,
        'backward': 90,
        'left':     45,
        'right':    135,
        'stop':     90,
    }
    servo_angle = angle_map.get(raw_dir, 90)
    dc_dir = raw_dir if raw_dir in ('forward', 'backward', 'stop') else 'stop'
    if raw_dir in ('left', 'right'):
        dc_dir = 'forward'  # turn while driving forward

    set_servo(servo_angle)
    set_dc_motor(speed, dc_dir)
    print(f"[MOTOR] dir={raw_dir} speed={speed}% servo={servo_angle}°")

# ── MEMORY ───────────────────────────────────────────────────────────────────
MEMORY_PATH = os.path.join(os.path.dirname(__file__), 'memory.log')

def load_memory() -> list[str]:
    if not os.path.exists(MEMORY_PATH):
        return []
    with open(MEMORY_PATH, 'r') as f:
        return [l.strip() for l in f if l.strip()]

def save_memory(entries: list[str]):
    with open(MEMORY_PATH, 'w') as f:
        f.write('\n'.join(entries) + '\n')

def append_memory(note: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    entry = f"[{ts}] {note}"
    with open(MEMORY_PATH, 'a') as f:
        f.write(entry + '\n')
    return entry

# ── ANTHROPIC CLIENT ─────────────────────────────────────────────────────────
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

SYSTEM_PROMPT = """You are the mind of a small wheeled robot with a camera.
You perceive the world through a camera image and a short memory log.
Respond ONLY with a valid JSON object — no prose, no markdown, no explanation.
The JSON must have exactly these keys:
  say   (string): What you want to say aloud. Be concise, max 20 words. Empty string if silent.
  move  (object): { "dir": "forward"|"backward"|"left"|"right"|"stop", "speed": 0-100 }
  memo  (string): One short fact to store in memory about what you just observed. Empty string if nothing notable.

Rules:
- Avoid obstacles. Be curious but safe.
- Move slowly in unfamiliar territory (speed 20-40%).
- Only move fast (60%+) when the path is clearly open.
- Use memory to avoid repeating mistakes.
- Keep 'say' natural and expressive — you have a personality."""

def query_claude(image_b64: str | None, memory_context: str, cycle: int) -> dict:
    content = []

    if image_b64:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_b64,
            }
        })

    mem_snippet = memory_context[-2000:] if memory_context else "(empty)"
    content.append({
        "type": "text",
        "text": f"Cycle #{cycle}. Recent memory:\n{mem_snippet}\n\nWhat do you observe? Respond with JSON only."
    })

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}]
    )

    raw = response.content[0].text.strip()
    # Strip any accidental markdown fences
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)

# ── FLASK APP ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='.')

@app.route('/')
def index():
    return send_from_directory('.', 'lepotato_dashboard.html')

@app.route('/api/think', methods=['POST'])
def think():
    data = request.get_json(force=True)
    image_b64     = data.get('image')       # may be None
    memory_ctx    = data.get('memory', '')
    cycle         = data.get('cycle', 0)

    try:
        result = query_claude(image_b64, memory_ctx, cycle)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Apply motor commands in a background thread (non-blocking)
    if 'move' in result:
        threading.Thread(target=apply_move_command, args=(result['move'],), daemon=True).start()

    # Persist memory note
    if result.get('memo'):
        append_memory(result['memo'])

    return jsonify(result)

@app.route('/api/cleanup_memory', methods=['POST'])
def cleanup_memory():
    data = request.get_json(force=True)
    entries = data.get('entries', [])

    try:
        cleaned = _ai_cleanup_memory(entries)
        save_memory(cleaned)
        return jsonify({"cleaned": cleaned})
    except Exception as e:
        return jsonify({"error": str(e), "cleaned": entries}), 200

def _ai_cleanup_memory(entries: list[str]) -> list[str]:
    """Call Claude to deduplicate and clean the memory log."""
    if len(entries) < 5:
        return entries

    blob = '\n'.join(entries[-100:])  # keep last 100 max
    prompt = f"""You are a memory curator for a robot. 
Below is a raw memory log with timestamps. Clean it:
1. Remove exact or near-duplicate entries.
2. Remove contradicted facts (keep the newer one).
3. Remove vague or uninformative entries.
4. Return ONLY the cleaned entries as a JSON array of strings. No commentary.

Memory log:
{blob}"""

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = resp.content[0].text.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    cleaned = json.loads(raw)
    return cleaned if isinstance(cleaned, list) else entries

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    setup_gpio()
    print("[SERVER] Starting on http://0.0.0.0:5000")
    print("[SERVER] Set ANTHROPIC_API_KEY in environment before running")
    app.run(host='0.0.0.0', port=5000, threaded=True)
