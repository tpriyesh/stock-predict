#!/bin/bash
# ============================================================
# Hostinger VPS Security Hardening
# ============================================================
# Run ONCE after first SSH into a fresh VPS:
#   ssh root@your-vps-ip
#   bash scripts/secure_vps.sh
#
# What it does:
#   1. Creates non-root 'deploy' user with sudo + SSH keys
#   2. Hardens SSH (key-only, no root, custom port 2222)
#   3. Sets up UFW firewall (only SSH allowed)
#   4. Installs fail2ban (bans brute-force attempts)
#   5. Enables automatic security updates
#   6. Sets strict file permissions
#
# After running, SSH access changes to:
#   ssh -p 2222 deploy@your-vps-ip
#
# IMPORTANT: Keep your current SSH session open until you
# verify the new setup works in a second terminal!
# ============================================================
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[+]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Must run as root
if [ "$EUID" -ne 0 ]; then
    error "Run this script as root: sudo bash $0"
fi

SSH_PORT="${SSH_PORT:-2222}"
DEPLOY_USER="${DEPLOY_USER:-deploy}"
PROJECT_DIR="${PROJECT_DIR:-/opt/stock-predict}"

echo ""
echo "============================================================"
echo "  VPS Security Hardening for Stock-Predict"
echo "============================================================"
echo "  SSH Port:    $SSH_PORT"
echo "  Deploy User: $DEPLOY_USER"
echo "  Project Dir: $PROJECT_DIR"
echo "============================================================"
echo ""

# ============================================================
# 1. SYSTEM UPDATES
# ============================================================
log "Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq

# ============================================================
# 2. CREATE NON-ROOT USER
# ============================================================
if id "$DEPLOY_USER" &>/dev/null; then
    log "User '$DEPLOY_USER' already exists"
else
    log "Creating user '$DEPLOY_USER'..."
    adduser --disabled-password --gecos "" "$DEPLOY_USER"
    usermod -aG sudo "$DEPLOY_USER"
    # Allow sudo without password for deploy user
    echo "$DEPLOY_USER ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/$DEPLOY_USER"
    chmod 440 "/etc/sudoers.d/$DEPLOY_USER"
fi

# Copy SSH keys from root to deploy user
if [ -f /root/.ssh/authorized_keys ]; then
    log "Copying SSH keys to $DEPLOY_USER..."
    mkdir -p "/home/$DEPLOY_USER/.ssh"
    cp /root/.ssh/authorized_keys "/home/$DEPLOY_USER/.ssh/"
    chown -R "$DEPLOY_USER:$DEPLOY_USER" "/home/$DEPLOY_USER/.ssh"
    chmod 700 "/home/$DEPLOY_USER/.ssh"
    chmod 600 "/home/$DEPLOY_USER/.ssh/authorized_keys"
else
    warn "No SSH keys found at /root/.ssh/authorized_keys"
    warn "You MUST add your SSH public key before disabling password auth!"
    warn "Run: ssh-copy-id -p $SSH_PORT $DEPLOY_USER@this-server"
fi

# ============================================================
# 3. SSH HARDENING
# ============================================================
log "Hardening SSH configuration..."

# Backup original config
cp /etc/ssh/sshd_config /etc/ssh/sshd_config.bak

# Apply hardened settings
cat > /etc/ssh/sshd_config.d/hardened.conf << 'SSHEOF'
# Stock-Predict VPS SSH Hardening
# Applied by secure_vps.sh

# Non-standard port (reduces automated scans by 99%)
Port SSH_PORT_PLACEHOLDER

# Authentication
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
AuthenticationMethods publickey
PermitEmptyPasswords no
MaxAuthTries 3

# Session
ClientAliveInterval 300
ClientAliveCountMax 2
LoginGraceTime 30

# Security
X11Forwarding no
AllowTcpForwarding no
AllowAgentForwarding no
PermitUserEnvironment no

# Only allow deploy user
AllowUsers DEPLOY_USER_PLACEHOLDER
SSHEOF

# Replace placeholders
sed -i "s/SSH_PORT_PLACEHOLDER/$SSH_PORT/g" /etc/ssh/sshd_config.d/hardened.conf
sed -i "s/DEPLOY_USER_PLACEHOLDER/$DEPLOY_USER/g" /etc/ssh/sshd_config.d/hardened.conf

# Also set Port in main config if sshd_config.d is not supported
if ! grep -q "^Include /etc/ssh/sshd_config.d/" /etc/ssh/sshd_config 2>/dev/null; then
    sed -i "s/^#\?Port .*/Port $SSH_PORT/" /etc/ssh/sshd_config
    sed -i "s/^#\?PermitRootLogin .*/PermitRootLogin no/" /etc/ssh/sshd_config
    sed -i "s/^#\?PasswordAuthentication .*/PasswordAuthentication no/" /etc/ssh/sshd_config
    sed -i "s/^#\?PubkeyAuthentication .*/PubkeyAuthentication yes/" /etc/ssh/sshd_config
fi

# Validate SSH config before restarting
sshd -t || error "SSH config validation failed! Check /etc/ssh/sshd_config.d/hardened.conf"

# ============================================================
# 4. UFW FIREWALL
# ============================================================
log "Setting up UFW firewall..."

apt-get install -y -qq ufw

# Reset and configure
ufw --force reset
ufw default deny incoming
ufw default allow outgoing

# Allow only SSH on custom port
ufw allow "$SSH_PORT/tcp" comment "SSH"

# Enable firewall
ufw --force enable
log "Firewall active. Only port $SSH_PORT (SSH) is open."

# ============================================================
# 5. FAIL2BAN
# ============================================================
log "Installing fail2ban..."

apt-get install -y -qq fail2ban

# Configure jail for SSH
cat > /etc/fail2ban/jail.local << JAILEOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5
backend = systemd

[sshd]
enabled = true
port = $SSH_PORT
filter = sshd
logpath = /var/log/auth.log
maxretry = 5
bantime = 3600
JAILEOF

systemctl enable fail2ban
systemctl restart fail2ban
log "Fail2ban active: 5 failed attempts = 1 hour ban"

# ============================================================
# 6. AUTOMATIC SECURITY UPDATES
# ============================================================
log "Enabling automatic security updates..."

apt-get install -y -qq unattended-upgrades apt-listchanges

cat > /etc/apt/apt.conf.d/50unattended-upgrades << 'UPGEOF'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}";
    "${distro_id}:${distro_codename}-security";
    "${distro_id}ESMApps:${distro_codename}-apps-security";
    "${distro_id}ESM:${distro_codename}-infra-security";
};
Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
// Auto-reboot at 4 AM IST (22:30 UTC) if needed — market is closed
Unattended-Upgrade::Automatic-Reboot "true";
Unattended-Upgrade::Automatic-Reboot-Time "22:30";
UPGEOF

cat > /etc/apt/apt.conf.d/20auto-upgrades << 'AUTOEOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
AUTOEOF

systemctl enable unattended-upgrades
log "Auto-updates enabled (security patches applied daily)"

# ============================================================
# 7. FILE PERMISSIONS
# ============================================================
if [ -d "$PROJECT_DIR" ]; then
    log "Setting file permissions in $PROJECT_DIR..."

    # .env — owner read/write only
    if [ -f "$PROJECT_DIR/.env" ]; then
        chmod 600 "$PROJECT_DIR/.env"
        log ".env: 600 (owner read/write only)"
    fi

    # data directory
    if [ -d "$PROJECT_DIR/data" ]; then
        chmod 700 "$PROJECT_DIR/data"
        log "data/: 700 (owner only)"
    fi

    # Zerodha token file
    if [ -f "$HOME/.zerodha_token.json" ]; then
        chmod 600 "$HOME/.zerodha_token.json"
        log ".zerodha_token.json: 600 (owner only)"
    fi

    # Ensure trading user owns the project
    if id "trading" &>/dev/null; then
        chown -R trading:trading "$PROJECT_DIR/data" "$PROJECT_DIR/logs" 2>/dev/null || true
    fi
else
    warn "Project directory $PROJECT_DIR not found. Set permissions manually after deployment."
fi

# ============================================================
# 8. MISC HARDENING
# ============================================================
log "Applying additional hardening..."

# Disable unused services
systemctl disable --now avahi-daemon 2>/dev/null || true
systemctl disable --now cups 2>/dev/null || true

# Restrict kernel parameters
cat > /etc/sysctl.d/99-hardening.conf << 'SYSEOF'
# Disable IP forwarding
net.ipv4.ip_forward = 0
net.ipv6.conf.all.forwarding = 0

# Ignore ICMP redirects
net.ipv4.conf.all.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv4.conf.all.send_redirects = 0

# Ignore source-routed packets
net.ipv4.conf.all.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0

# Enable SYN flood protection
net.ipv4.tcp_syncookies = 1

# Log suspicious packets
net.ipv4.conf.all.log_martians = 1

# Ignore ICMP broadcast requests
net.ipv4.icmp_echo_ignore_broadcasts = 1

# Protect against time-wait assassination
net.ipv4.tcp_rfc1337 = 1
SYSEOF

sysctl -p /etc/sysctl.d/99-hardening.conf > /dev/null 2>&1

# Set timezone to IST
timedatectl set-timezone Asia/Kolkata

# ============================================================
# 9. INSTALL DOCKER (if not present)
# ============================================================
if ! command -v docker &>/dev/null; then
    log "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker "$DEPLOY_USER"
    systemctl enable docker
    log "Docker installed. $DEPLOY_USER added to docker group."
else
    log "Docker already installed"
    usermod -aG docker "$DEPLOY_USER" 2>/dev/null || true
fi

# ============================================================
# RESTART SSH (must be last)
# ============================================================
warn ""
warn "============================================================"
warn "  IMPORTANT: TEST BEFORE CLOSING THIS SESSION!"
warn "============================================================"
warn ""
warn "  1. Open a NEW terminal"
warn "  2. Test SSH: ssh -p $SSH_PORT $DEPLOY_USER@$(hostname -I | awk '{print $1}')"
warn "  3. If it works, close this session"
warn "  4. If it fails, fix from THIS session"
warn ""
warn "============================================================"
warn ""

read -p "Press Enter to restart SSH and apply changes... " _

systemctl restart sshd

log ""
log "============================================================"
log "  VPS HARDENED SUCCESSFULLY"
log "============================================================"
log "  SSH:        port $SSH_PORT, key-only, no root"
log "  Firewall:   only port $SSH_PORT open"
log "  Fail2ban:   5 attempts = 1 hour ban"
log "  Updates:    automatic security patches"
log "  Timezone:   Asia/Kolkata (IST)"
log "  Docker:     installed, $DEPLOY_USER in docker group"
log ""
log "  Next steps:"
log "    1. Test SSH in new terminal (see above)"
log "    2. Deploy code: bash scripts/deploy.sh $DEPLOY_USER@$(hostname -I | awk '{print $1}')"
log "    3. Copy .env to server"
log "    4. Start trading: docker compose up -d"
log "============================================================"
