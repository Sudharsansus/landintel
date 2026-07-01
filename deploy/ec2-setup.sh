#!/usr/bin/env bash
# EC2 first-boot setup for LandIntel on Ubuntu 22.04 LTS (t3.medium).
# Run once as root (or via cloud-init UserData) immediately after instance launch.
# Safe to re-run -- all steps are idempotent.
set -euo pipefail

APP_DIR=/opt/landintel
APP_USER=ec2-user  # Amazon Linux; change to "ubuntu" on Ubuntu AMI

# ── System packages ─────────────────────────────────────────────────────────
apt-get update -y
apt-get install -y --no-install-recommends \
    docker.io \
    docker-compose-plugin \
    nginx \
    git \
    curl

# ── Docker ──────────────────────────────────────────────────────────────────
systemctl enable --now docker
usermod -aG docker "$APP_USER"

# ── Application directory ────────────────────────────────────────────────────
mkdir -p "$APP_DIR"
chown "$APP_USER":"$APP_USER" "$APP_DIR"
# The repo should be cloned / rsync'd here by CI before systemd starts the services.

# ── Nginx ───────────────────────────────────────────────────────────────────
cp "$APP_DIR/deploy/nginx.conf" /etc/nginx/sites-available/landintel
ln -sf /etc/nginx/sites-available/landintel /etc/nginx/sites-enabled/landintel
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl enable --now nginx

# ── Frontend static files ────────────────────────────────────────────────────
# Built by CI (`npm run build` in frontend/) and rsync'd to the instance.
# nginx serves them from /var/www/landintel.
mkdir -p /var/www/landintel
chown "$APP_USER":"$APP_USER" /var/www/landintel

# ── Systemd services ─────────────────────────────────────────────────────────
cp "$APP_DIR/deploy/systemd/landintel-api.service"    /etc/systemd/system/
cp "$APP_DIR/deploy/systemd/landintel-worker.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable landintel-api landintel-worker

echo ""
echo "Setup complete. To start the stack:"
echo "  cd $APP_DIR && docker compose pull && docker compose up -d"
echo "  systemctl start landintel-api landintel-worker"
