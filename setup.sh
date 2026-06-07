#!/bin/bash
# Le Potato AI Body — Ubuntu 22.04 LTS setup script
# AML-S905X-CC target
#
# Usage:
#   bash setup.sh           — full setup (Le Potato)
#   bash setup.sh --pc      — PC-side offload server only (Linux desktop)
#   bash setup.sh --help    — show this message
#
# What this does:
#   - Installs system + Python deps
#   - Replaces pigpio with lgpio (correct GPIO library for Le Potato)
#   - Generates a self-signed TLS cert (needed for webcam in browser)
#   - Installs and starts Ollama with the required models
#   - Creates a systemd service so everything starts on boot

set -e

# ── ARGS ──────────────────────────────────────────────────────────────────────
MODE="lepotato"
for arg in "$@"; do
  case "$arg" in
    --pc)   MODE="pc" ;;
    --help) grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
  esac
done

# ── COLOURS ───────────────────────────────────────────────────────────────────
G='\033[0;32m'; Y='\033[0;33m'; R='\033[0;31m'; B='\033[0;34m'; N='\033[0m'
info()  { echo -e "${B}[INFO]${N}  $*"; }
ok()    { echo -e "${G}[ OK ]${N}  $*"; }
warn()  { echo -e "${Y}[WARN]${N}  $*"; }
error() { echo -e "${R}[ERR ]${N}  $*"; }
step()  { echo -e "\n${G}━━━ $* ${N}"; }

# ── DETECT USER ───────────────────────────────────────────────────────────────
REALUSER="${SUDO_USER:-$USER}"
REALHOME=$(eval echo "~$REALUSER")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo -e "${G}╔══════════════════════════════════════════╗"
echo    "║   Le Potato AI Body — Setup Script       ║"
echo -e "╚══════════════════════════════════════════╝${N}"
echo ""
info "Mode: $MODE"
info "User: $REALUSER  Home: $REALHOME"
info "Script dir: $SCRIPT_DIR"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# PC MODE — just Python deps + Ollama for the offload server
# ─────────────────────────────────────────────────────────────────────────────
if [ "$MODE" = "pc" ]; then
  step "PC offload server setup"

  info "Updating package lists…"
  sudo apt update -qq

  info "Installing Python deps…"
  pip3 install --break-system-packages flask requests

  # Ollama
  step "Ollama"
  if command -v ollama &>/dev/null; then
    ok "Ollama already installed: $(ollama --version 2>/dev/null || echo '?')"
  else
    info "Installing Ollama…"
    curl -fsSL https://ollama.com/install.sh | sh
    ok "Ollama installed"
  fi

  info "Starting Ollama service…"
  if systemctl is-active --quiet ollama 2>/dev/null; then
    ok "Ollama service already running"
  else
    sudo systemctl enable ollama
    sudo systemctl start ollama
    sleep 2
  fi

  step "Pulling AI models"
  info "Pulling moondream (vision, ~1.7 GB)…"
  ollama pull moondream
  info "Pulling llama3.2:3b (text/decision, ~2 GB)…"
  ollama pull llama3.2:3b
  ok "Models ready"

  # Firewall hint
  step "Firewall"
  if command -v ufw &>/dev/null && sudo ufw status | grep -q "Status: active"; then
    warn "ufw is active — opening port 11435 for Le Potato…"
    sudo ufw allow 11435/tcp
    ok "Port 11435 open"
  else
    info "ufw not active — no firewall rule needed"
  fi

  # Print connection info
  LAN_IP=$(hostname -I | awk '{print $1}')
  step "Done — PC offload server"
  echo ""
  echo -e "  Start the offload server:"
  echo -e "    ${G}python3 $SCRIPT_DIR/ai_server.py${N}"
  echo -e "  Or with debug logging:"
  echo -e "    ${G}python3 $SCRIPT_DIR/ai_server.py --debug${N}"
  echo ""
  echo -e "  On Le Potato, set this in server.py:"
  echo -e "    ${Y}OFFLOAD_URL = \"http://$LAN_IP:11435\"${N}"
  echo ""
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# LE POTATO MODE — full setup
# ─────────────────────────────────────────────────────────────────────────────

# Must be run as root / sudo for GPIO and systemd steps
if [ "$EUID" -ne 0 ]; then
  warn "Some steps require root. Re-running with sudo…"
  exec sudo bash "$0" "$@"
fi

# ── 1. System packages ────────────────────────────────────────────────────────
step "System packages"
apt update -qq
apt install -y \
  python3-pip \
  python3-dev \
  python3-libgpiod \
  libretech-gpio \
  fswebcam \
  git \
  curl \
  openssl

# Remove pigpio if previously installed — wrong library for Le Potato
if dpkg -l pigpio &>/dev/null 2>&1; then
  warn "Removing pigpio (not compatible with AML-S905X-CC)…"
  apt remove -y pigpio python3-pigpio 2>/dev/null || true
fi

ok "System packages installed"

# ── 2. Python packages ────────────────────────────────────────────────────────
step "Python packages"
pip3 install --break-system-packages \
  flask \
  lgpio \
  requests \
  pillow

ok "Python packages installed"

# ── 3. lgpio permissions ──────────────────────────────────────────────────────
step "GPIO permissions"
# Le Potato doesn't have a 'gpio' group by default — we use udev rules instead
UDEV_RULE='/etc/udev/rules.d/99-lepotato-gpio.rules'
if [ ! -f "$UDEV_RULE" ]; then
  info "Creating udev rule for gpiochip access…"
  cat > "$UDEV_RULE" << 'UDEV'
# Le Potato — allow group 'dialout' to access gpiochip devices
SUBSYSTEM=="gpio", KERNEL=="gpiochip*", GROUP="dialout", MODE="0660"
UDEV
  udevadm control --reload-rules
  udevadm trigger
  ok "udev rule created: $UDEV_RULE"
else
  ok "udev rule already exists"
fi

# Add the real user to dialout so sudo isn't needed for GPIO
if id -nG "$REALUSER" | grep -qw dialout; then
  ok "$REALUSER already in dialout group"
else
  usermod -aG dialout "$REALUSER"
  warn "$REALUSER added to dialout — log out and back in for GPIO without sudo"
fi

# ── 4. TLS certificate ────────────────────────────────────────────────────────
step "TLS certificate (needed for webcam in browser)"
CERT_DIR="$SCRIPT_DIR"
if [ -f "$CERT_DIR/cert.pem" ] && [ -f "$CERT_DIR/key.pem" ]; then
  ok "cert.pem / key.pem already exist — skipping"
else
  LAN_IP=$(hostname -I | awk '{print $1}')
  HOSTNAME=$(hostname)
  info "Generating self-signed cert for $HOSTNAME / $LAN_IP …"
  openssl req -x509 -newkey rsa:2048 \
    -keyout "$CERT_DIR/key.pem" \
    -out    "$CERT_DIR/cert.pem" \
    -days 3650 -nodes \
    -subj "/CN=$HOSTNAME" \
    -addext "subjectAltName=IP:$LAN_IP,DNS:$HOSTNAME,DNS:lepotato.local" \
    2>/dev/null
  chown "$REALUSER":"$REALUSER" "$CERT_DIR/cert.pem" "$CERT_DIR/key.pem"
  ok "Certificate generated: $CERT_DIR/cert.pem"
  warn "You'll need to accept the browser security warning on first visit"
fi

# ── 5. Ollama ─────────────────────────────────────────────────────────────────
step "Ollama"
# Only install if not offloading — but we install anyway since local fallback
# is useful. Models won't be pulled if OFFLOAD_URL is set in server.py.
if command -v ollama &>/dev/null; then
  ok "Ollama already installed: $(ollama --version 2>/dev/null || echo '?')"
else
  info "Installing Ollama…"
  curl -fsSL https://ollama.com/install.sh | sh
  ok "Ollama installed"
fi

if systemctl is-active --quiet ollama 2>/dev/null; then
  ok "Ollama service already running"
else
  systemctl enable ollama
  systemctl start ollama
  sleep 2
fi

# ── 6. Pull models (skip if offload URL is set in server.py) ─────────────────
step "AI models"
OFFLOAD_SET=$(grep -E "^OFFLOAD_URL\s*=\s*\"http" "$SCRIPT_DIR/server.py" 2>/dev/null || true)

if [ -n "$OFFLOAD_SET" ]; then
  info "OFFLOAD_URL is configured in server.py — skipping local model pull"
  info "Models will be served by your PC running ai_server.py"
else
  info "Pulling moondream (vision model, ~1.7 GB)…"
  ollama pull moondream
  info "Pulling llama3.2:3b (decision model, ~2 GB)…"
  ollama pull llama3.2:3b
  ok "Models ready"
fi

# ── 7. systemd service ────────────────────────────────────────────────────────
step "systemd service (auto-start on boot)"
SERVICE_FILE='/etc/systemd/system/lepotato-ai.service'
cat > "$SERVICE_FILE" << SYSTEMD
[Unit]
Description=Le Potato AI Body Server
After=network-online.target ollama.service
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=$SCRIPT_DIR
ExecStart=/usr/bin/python3 $SCRIPT_DIR/server.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
SYSTEMD

systemctl daemon-reload
systemctl enable lepotato-ai.service
ok "Service installed: lepotato-ai.service"
info "Commands: sudo systemctl start|stop|status|logs lepotato-ai"
info "Logs:     sudo journalctl -u lepotato-ai -f"

# ── 8. Verify GPIO lines ──────────────────────────────────────────────────────
step "GPIO pin verification"
info "Checking expected lines via lgpio info…"
declare -A PINS=( [12]="DC PWM" [16]="DC dir1" [18]="DC dir2" [33]="Servo" )
ALL_OK=1
for PIN in "${!PINS[@]}"; do
  RESULT=$(lgpio info "$PIN" gpiod 2>/dev/null || echo "ERROR")
  if [ "$RESULT" = "ERROR" ] || [ -z "$RESULT" ]; then
    warn "Pin $PIN (${PINS[$PIN]}): could not read — is libretech-gpio installed?"
    ALL_OK=0
  else
    CHIP=$(echo "$RESULT" | awk '{print $1}')
    LINE=$(echo "$RESULT" | awk '{print $2}')
    ok "Pin $PIN (${PINS[$PIN]}): chip=$CHIP line=$LINE"
  fi
done
if [ "$ALL_OK" = "0" ]; then
  warn "Some pins couldn't be verified. Make sure libretech-gpio is installed and you're on a Libre Computer image."
fi

# ── 9. Summary ────────────────────────────────────────────────────────────────
LAN_IP=$(hostname -I | awk '{print $1}')

step "Setup complete!"
echo ""
echo -e "  ${G}Next steps:${N}"
echo ""

if [ -n "$OFFLOAD_SET" ]; then
  echo -e "  1. On your PC, run:"
  echo -e "       ${G}python3 ai_server.py${N}"
  echo -e "  2. Start the Le Potato server:"
  echo -e "       ${G}sudo systemctl start lepotato-ai${N}"
  echo -e "     Or manually:"
  echo -e "       ${G}sudo python3 $SCRIPT_DIR/server.py${N}"
else
  echo -e "  1. Start the server:"
  echo -e "       ${G}sudo systemctl start lepotato-ai${N}"
  echo -e "     Or manually:"
  echo -e "       ${G}sudo python3 $SCRIPT_DIR/server.py${N}"
  echo ""
  echo -e "  To offload AI to your PC later:"
  echo -e "    - Run ${Y}bash setup.sh --pc${N} on the PC"
  echo -e "    - Set ${Y}OFFLOAD_URL${N} in server.py to your PC's IP"
fi

echo ""
echo -e "  Dashboard:   ${G}https://$LAN_IP:5000${N}"
echo -e "  Debug log:   ${G}https://$LAN_IP:5000/api/debug${N}"
echo -e "  Status:      ${G}https://$LAN_IP:5000/api/status${N}"
echo ""
if id -nG "$REALUSER" | grep -qw dialout; then
  true
else
  warn "Log out and back in as $REALUSER to use GPIO without sudo"
fi
echo ""
