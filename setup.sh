#!/bin/bash
# Le Potato AI Body — Ubuntu 22.04 LTS setup script
# Run once after flashing the OS:  bash setup.sh

set -e
echo "=== Le Potato AI Body — Setup ==="

# System packages
sudo apt update
sudo apt install -y \
  python3-pip python3-dev \
  pigpio               \  # GPIO daemon (works on AML-S905X-CC)
  python3-pigpio       \
  fswebcam             \  # test webcam from CLI
  git curl

# Python packages
pip3 install --break-system-packages \
  flask \
  anthropic \
  pillow

# Enable pigpio daemon on boot
sudo systemctl enable pigpiod
sudo systemctl start pigpiod

# Set your API key (edit this line)
echo "export ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE" >> ~/.bashrc
echo ""
echo "=== Done! ==="
echo "Edit ~/.bashrc and add your real ANTHROPIC_API_KEY, then:"
echo "  source ~/.bashrc"
echo "  python3 server.py"
echo "  Open http://$(hostname -I | awk '{print $1}'):5000 in a browser"
