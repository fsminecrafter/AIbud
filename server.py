#!/usr/bin/env python3
"""
Le Potato AI Body — backend server
Ubuntu 22.04 LTS target (AML-S905X-CC / Le Potato)

Install deps:
  pip3 install flask anthropic lgpio pillow --break-system-packages
  sudo apt install -y gpiod openssl

GPIO wiring (Le Potato AML-S905X-CC 40-pin header):
  Servo signal  → physical pin 33  (gpiochip0 line 10  — AO_10)
  DC motor PWM  → physical pin 12  (gpiochip1 line 116 — GPIOX_12)
  DC motor dir1 → physical pin 16  (gpiochip1 line 118 — GPIOX_14? run gpioinfo to confirm)
  DC motor dir2 → physical pin 18  (gpiochip1 line 119)
  GND           → pins 6, 9, 14...

Run:
  # Generate self-signed cert once (required for HTTPS / camera access):
  openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 3650 -nodes -subj "/CN=lepotato.local"

  export ANTHROPIC_API_KEY=sk-ant-...
  python3 server.py
  Open browser → https://<lepotato-ip>:5000  (accept the cert warning)

To find your GPIO line numbers:
  sudo apt install gpiod
  gpiodetect           # lists gpiochip0, gpiochip1...
  gpioinfo gpiochip1   # shows line numbers and current names
"""

import json
import os
import time
import threading

from flask import Flask, request, jsonify, send_from_directory
import anthropic

# ── GPIO via lgpio (no daemon needed, works on AML-S905X-CC) ─────────────────
#
# lgpio addresses GPIO lines as (chip_handle, line_offset).
# Run `gpioinfo gpiochip0` and `gpioinfo gpiochip1` to find your line offsets.
# The values below are best-guess for Le Potato — VERIFY with gpioinfo first.
#
# Physical pin → chip/line mapping (AML-S905X-CC):
#   pin 33 → gpiochip0 line 10   (AO domain)
#   pin 12 → gpiochip1 line 116  (GPIOX_12)
#   pin 16 → gpiochip1 line 118  (GPIOX_14 — double-check)
#   pin 18 → gpiochip1 line 119  (GPIOX_15 — double-check)

CHIP_AO   = 0   # gpiochip0 — AO (always-on) domain — pin 33 lives here
CHIP_MAIN = 1   # gpiochip1 — main GPIO bank

LINE_SERVO    = (CHIP_AO,   10)   # physical pin 33
LINE_DC_PWM   = (CHIP_MAIN, 116)  # physical pin 12
LINE_DC_DIR1  = (CHIP_MAIN, 118)  # physical pin 16
LINE_DC_DIR2  = (CHIP_MAIN, 119)  # physical pin 18

SERVO_MIN_PW  = 500    # microseconds — hard left  (~0°)
SERVO_MAX_PW  = 2500   # microseconds — hard right (~180°)
SERVO_CENTER  = 1500   # microseconds — center     (90°)
SERVO_HZ      = 50     # standard servo PWM frequency

try:
    import lgpio
    _h = {}  # chip_index → handle

    def _chip(idx):
        if idx not in _h:
            _h[idx] = lgpio.gpiochip_open(idx)
        return _h[idx]

    # Claim all lines as outputs
    lgpio.gpio_claim_output(_chip(LINE_SERVO[0]),   LINE_SERVO[1])
    lgpio.gpio_claim_output(_chip(LINE_DC_PWM[0]),  LINE_DC_PWM[1])
    lgpio.gpio_claim_output(_chip(LINE_DC_DIR1[0]), LINE_DC_DIR1[1])
    lgpio.gpio_claim_output(_chip(LINE_DC_DIR2[0]), LINE_DC_DIR2[1])

    GPIO_AVAILABLE = True
    print("[GPIO] lgpio ready")

except Exception as e:
    print(f"[GPIO] Not available ({e}). Motor commands will be logged only.")
    GPIO_AVAILABLE = False

# ── SOFTWARE PWM HELPERS ─────────────────────────────────────────────────────
# lgpio provides tx_pwm for hardware-assisted PWM on supported lines,
# and gpio_write for simple digital output.

def _write(line_tuple, value: int):
    if not GPIO_AVAILABLE:
        return
    chip, line = line_tuple
    lgpio.gpio_write(_chip(chip), line, value)

def _pwm(line_tuple, freq: int, duty_pct: float):
    """Software PWM via lgpio tx_pwm. duty_pct: 0.0–100.0"""
    if not GPIO_AVAILABLE:
        return
    chip, line = line_tuple
    if duty_pct <= 0:
        lgpio.tx_pwm(_chip(chip), line, freq, 0)
    else:
        lgpio.tx_pwm(_chip(chip), line, freq, max(0.0, min(100.0, duty_pct)))

def setup_gpio():
    if not GPIO_AVAILABLE:
        return
    # Centre servo on startup
    set_servo(90)
    set_dc_motor(0, 'stop')
    print("[GPIO] Servo centred, DC motor stopped.")

def set_servo(angle_deg: float):
    """angle_deg: 0–180. 90 = straight ahead."""
    if not GPIO_AVAILABLE:
        return
    angle_deg = max(0.0, min(180.0, angle_deg))
    pw_us = SERVO_MIN_PW + (angle_deg / 180.0) * (SERVO_MAX_PW - SERVO_MIN_PW)
    # Convert pulse width to duty cycle at 50 Hz (period = 20 000 µs)
    duty = (pw_us / 20_000.0) * 100.0
    _pwm(LINE_SERVO, SERVO_HZ, duty)

def set_dc_motor(speed_pct: float, direction: str):
    """speed_pct: 0–100. direction: 'forward' | 'backward' | 'stop'."""
    if not GPIO_AVAILABLE:
        return
    speed_pct = max(0.0, min(100.0, speed_pct))

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

def apply_move_command(move: dict):
    raw_dir = move.get('dir', 'stop').lower()
    speed   = float(move.get('speed', 0))

    angle_map = {
        'forward':  90,
        'backward': 90,
        'left':     45,
        'right':    135,
        'stop':     90,
    }
    servo_angle = angle_map.get(raw_dir, 90)
    dc_dir = raw_dir if raw_dir in ('forward', 'backward', 'stop') else 'forward'

    set_servo(servo_angle)
    set_dc_motor(speed, dc_dir)
    print(f"[MOTOR] dir={raw_dir} speed={speed:.0f}% servo={servo_angle}°")

# ── MEMORY ───────────────────────────────────────────────────────────────────
MEMORY_PATH = os.path.join(os.path.dirname(__file__), 'memory.log')

def load_memory() -> list:
    if not os.path.exists(MEMORY_PATH):
        return []
    with open(MEMORY_PATH, 'r') as f:
        return [l.strip() for l in f if l.strip()]

def save_memory(entries: list):
    with open(MEMORY_PATH, 'w') as f:
        f.write('\n'.join(entries) + '\n')

def append_memory(note: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    entry = f"[{ts}] {note}"
    with open(MEMORY_PATH, 'a') as f:
        f.write(entry + '\n')
    return entry

# ── ANTHROPIC CLIENT ─────────────────────────────────────────────────────────
client = anthropic.Anthropic()

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

def query_claude(image_b64, memory_context: str, cycle: int) -> dict:
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
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)

# ── FLASK APP ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='.')

@app.route('/')
def index():
    return send_from_directory('.', 'lepotato_dashboard.html')

@app.route('/api/think', methods=['POST'])
def think():
    data      = request.get_json(force=True)
    image_b64 = data.get('image')
    memory_ctx = data.get('memory', '')
    cycle     = data.get('cycle', 0)

    try:
        result = query_claude(image_b64, memory_ctx, cycle)
    except Exception as e:
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
        cleaned = _ai_cleanup_memory(entries)
        save_memory(cleaned)
        return jsonify({"cleaned": cleaned})
    except Exception as e:
        return jsonify({"error": str(e), "cleaned": entries}), 200

def _ai_cleanup_memory(entries: list) -> list:
    if len(entries) < 5:
        return entries

    blob = '\n'.join(entries[-100:])
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

    # HTTPS is required for getUserMedia (camera) to work in browsers.
    # Generate cert once: openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 3650 -nodes -subj "/CN=lepotato.local"
    cert = ('cert.pem', 'key.pem')
    if os.path.exists('cert.pem') and os.path.exists('key.pem'):
        print("[SERVER] HTTPS enabled — open https://<ip>:5000")
        ssl_ctx = cert
    else:
        print("[SERVER] WARNING: No cert.pem/key.pem found. Camera will not work over plain HTTP.")
        print("[SERVER] Run: openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 3650 -nodes -subj '/CN=lepotato.local'")
        ssl_ctx = None

    print("[SERVER] Set ANTHROPIC_API_KEY in environment before running")
    app.run(host='0.0.0.0', port=5000, threaded=True, ssl_context=ssl_ctx)
