# Deployment & VPS Setup Guide

Reference document for the Hostinger VPS deployment. Covers what was done, why, challenges faced, and how to maintain it.

## Table of Contents

- [Infrastructure Overview](#infrastructure-overview)
- [VPS Setup](#vps-setup)
- [Security Hardening](#security-hardening)
- [Docker Deployment](#docker-deployment)
- [Git-Based Workflow](#git-based-workflow)
- [Challenges & Solutions](#challenges--solutions)
- [Daily Operations](#daily-operations)
- [Troubleshooting](#troubleshooting)

---

## Infrastructure Overview

### Why a VPS?

The trading agent runs 9:15 AM - 3:30 PM IST daily. Running on a laptop has a critical flaw: **laptop sleep = missed square-off**. If positions aren't exited before 3:30 PM, they carry overnight risk. A VPS runs 24/7 regardless of your laptop state.

### What We Chose

| Component | Choice | Why |
|-----------|--------|-----|
| **Provider** | Hostinger KVM 4 | Cost-effective, Mumbai data center (low latency to NSE), KVM virtualization (dedicated resources) |
| **Plan** | 16GB RAM, 4 vCPU, 200GB SSD | Headroom for trading agent + future services (bots, etc). Trading agent uses ~600MB RAM |
| **OS** | Ubuntu 24.04 LTS | Long-term support until 2029, best Docker support, our scripts are Debian/Ubuntu-based |
| **Deployment** | Docker Compose | Reproducible, isolated, easy rollback via git, read-only filesystem |

### Server Details

- **IP**: 72.62.230.66
- **SSH Port**: 2222 (non-standard to reduce scan noise)
- **User**: `deploy` (no root access)
- **Project Path**: `/opt/stock-predict`
- **Timezone**: Asia/Kolkata (IST)

---

## VPS Setup

### Step-by-Step (What Was Done)

#### 1. System Update
```bash
apt-get update && apt-get upgrade -y
```
**Why**: Fresh VPS images often have pending security patches. Always update before doing anything else.

#### 2. Created Non-Root User (`deploy`)
```bash
adduser --disabled-password deploy
usermod -aG sudo deploy
echo "deploy ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/deploy
```
**Why**: Running as root is dangerous — a compromised process gets full system access. The `deploy` user has sudo when needed but runs unprivileged by default. NOPASSWD avoids interactive prompts during automated deployments.

#### 3. SSH Key Authentication
```bash
# Copied SSH public key from root to deploy user
mkdir -p /home/deploy/.ssh
cp /root/.ssh/authorized_keys /home/deploy/.ssh/
chmod 700 /home/deploy/.ssh
chmod 600 /home/deploy/.ssh/authorized_keys
```
**Why**: SSH keys are cryptographically stronger than passwords. A 256-bit Ed25519 key is effectively unbrute-forceable, while passwords can be guessed.

---

## Security Hardening

### SSH Hardening (`/etc/ssh/sshd_config.d/hardened.conf`)

| Setting | Value | Why |
|---------|-------|-----|
| `Port 2222` | Non-standard port | Eliminates 99% of automated SSH brute-force scans (they target port 22) |
| `PermitRootLogin no` | Disabled | Even if root password leaks, SSH as root is impossible |
| `PasswordAuthentication no` | Key-only | Eliminates password brute-force entirely |
| `MaxAuthTries 3` | 3 attempts | Combined with fail2ban, limits attack surface |
| `AllowUsers deploy` | Whitelist | Only `deploy` can SSH in, no other users |
| `X11Forwarding no` | Disabled | No GUI forwarding needed — reduces attack surface |
| `AllowTcpForwarding no` | Disabled | Prevents SSH tunneling abuse |
| `ClientAliveInterval 300` | 5 min timeout | Drops idle connections, prevents abandoned sessions |
| `LoginGraceTime 30` | 30 seconds | Short window to authenticate, limits slow brute-force |

### Firewall (UFW)

```
Status: active
To                  Action      From
--                  ------      ----
2222/tcp            ALLOW IN    Anywhere    # SSH
```

**Why**: Default deny inbound + only SSH port open. The trading agent makes **outbound** API calls only (to brokers, news APIs, etc.) — it doesn't serve any web traffic. No HTTP/HTTPS ports needed.

**Best Practice**: Principle of least privilege — only open ports that are actively needed.

### Fail2Ban

```ini
[sshd]
enabled = true
port = 2222
maxretry = 5
bantime = 3600   # 1 hour ban
findtime = 600   # within 10 minutes
```

**Why**: Even with key-only auth, fail2ban adds defense-in-depth. After 5 failed attempts in 10 minutes, the IP is banned for 1 hour. This protects against:
- Scanners that hammer the port
- Key-based attacks (rare but possible)
- Resource exhaustion from connection floods

### Automatic Security Updates

```ini
Unattended-Upgrade::Automatic-Reboot "true";
Unattended-Upgrade::Automatic-Reboot-Time "22:30";  # 4 AM IST
```

**Why**: Unpatched servers are the #1 cause of VPS compromise. Auto-updates ensure security patches are applied within 24 hours. Reboot time is 4 AM IST — well after market close (3:30 PM) and before market open (9:15 AM).

### Kernel Hardening (`/etc/sysctl.d/99-hardening.conf`)

| Setting | Why |
|---------|-----|
| `tcp_syncookies = 1` | Protects against SYN flood DDoS |
| `accept_redirects = 0` | Prevents ICMP redirect attacks (MITM) |
| `log_martians = 1` | Logs suspicious packets for forensics |
| `ip_forward = 0` | This isn't a router — disable forwarding |
| `accept_source_route = 0` | Prevents source routing attacks |
| `tcp_rfc1337 = 1` | Protects against TIME-WAIT assassination |

### File Permissions

| File | Permission | Why |
|------|-----------|-----|
| `.env` | `600` (owner read/write) | Contains API keys, broker credentials — no other user should read it |
| `data/` | `700` (owner only) | Trade database has financial data |

### Docker Security

| Setting | Why |
|---------|-----|
| `read_only: true` | Container filesystem is immutable — malware can't modify application code |
| `tmpfs: /tmp, /app/__pycache__` | Only these paths are writable (in-memory, cleared on restart) |
| `cap_drop: ALL` | Drops all Linux capabilities — container can't mount filesystems, change network, etc. |
| `no-new-privileges: true` | Prevents privilege escalation via setuid binaries |
| Non-root user (`trading`) | Container runs as unprivileged user with `/usr/sbin/nologin` shell |
| `env_file: .env` | Secrets passed as env vars, not baked into the image |

---

## Docker Deployment

### Architecture

```
Host (Ubuntu 24.04)
├── Docker Engine
│   └── stock-predict container
│       ├── Python 3.11 (slim-bookworm)
│       ├── IST timezone configured
│       ├── Read-only filesystem
│       ├── Runs as 'trading' user (non-root)
│       └── Volumes:
│           ├── trading_data → /app/data (trades.db, reports)
│           └── trading_logs → /app/logs
├── /opt/stock-predict (git repo)
│   ├── .env (chmod 600, not in git)
│   ├── docker-compose.yml
│   └── docker-compose.live.yml
└── UFW firewall (port 2222 only)
```

### Docker DNS Fix

Docker containers use the host's DNS by default. On this VPS, the host uses `systemd-resolved` with a stub resolver at `127.0.0.53`. Docker containers can't reach this because it's a loopback address on the host's network namespace.

**Fix** (`/etc/docker/daemon.json`):
```json
{
    "dns": ["8.8.8.8", "1.1.1.1"]
}
```

This tells Docker to use Google DNS (8.8.8.8) and Cloudflare DNS (1.1.1.1) directly, bypassing the host's stub resolver.

### Why Docker (Not Just Python Directly)?

1. **Reproducibility**: Same Python version, same dependencies, same behavior on any machine
2. **Isolation**: Container can't affect host system even if compromised
3. **Easy rollback**: `docker compose down && git checkout <old-commit> && docker compose up -d`
4. **Read-only filesystem**: Application code can't be tampered with at runtime
5. **Clean restarts**: `docker compose restart` gives a fresh process without stale state

---

## Git-Based Workflow

### Why Git (Not rsync/scp)?

1. **Traceability**: Every change has a commit message, author, and timestamp
2. **Rollback**: `git revert` or `git checkout <commit>` to undo any change
3. **No drift**: VPS always matches exactly what's in the repo
4. **CI-ready**: Can add GitHub Actions later for automated testing before deploy
5. **Single source of truth**: GitHub repo is the canonical version

### Production Deployment Flow

```
Local Machine                    GitHub                     VPS
─────────────                    ──────                     ───
1. Edit code
2. Run tests locally
   pytest tests/ -v
3. git add + commit + push  ──→  Repo updated
4. ssh vps                                            ──→  Login
5.                                                         git pull
6.                                                         docker compose build
7.                                                         docker compose up -d
```

### Quick Deploy (One Command from Mac)

```bash
# Paper trading
ssh vps "cd /opt/stock-predict && git pull && docker compose build && docker compose up -d"

# Live trading (Zerodha)
ssh vps "cd /opt/stock-predict && git pull && docker compose build && docker compose -f docker-compose.yml -f docker-compose.live.yml up -d"

# Check it's running
ssh vps "docker logs -f stock-predict"
```

### SSH Aliases (Pre-configured in `~/.ssh/config`)

```
ssh vps          # Connect as deploy@72.62.230.66:2222 (daily use)
ssh vps-root     # Root access on port 22 (disabled after hardening)
```

### What's in Git vs What's NOT

| In Git | NOT in Git | Why |
|--------|-----------|-----|
| All source code | `.env` | Contains API keys |
| Dockerfile, compose files | `data/` | Runtime trade database |
| Test suite | `logs/` | Runtime logs |
| Scripts (deploy, secure) | `reference_repos/` | Third-party code, 54MB |
| `.env.example` (template) | `.claude/` | IDE/tool config |

### .gitignore Design

The `.gitignore` uses `/data/` (with leading slash) instead of `data/` to only exclude the top-level data directory. Without the slash, `src/data/` (source code) was also excluded — this was a bug we caught and fixed.

---

## Challenges & Solutions

### 1. SSH Lockout After Hardening

**Problem**: After setting UFW to allow only port 2222 and moving SSH to port 2222, we got locked out. Both ports timed out.

**Root Cause**: Two issues compounded:
- Ubuntu 24.04's `sshd_config.d/` include mechanism didn't override the main `sshd_config`'s `Port 22` setting
- Hostinger may have an external firewall layer beyond our UFW

**Solution**:
1. Used Hostinger's **web-based VPS console** (bypasses network firewall entirely — it's a serial console)
2. Added port 22 temporarily via `ufw allow 22/tcp`
3. Edited the main `sshd_config` directly: `sed -i 's/^#\?Port .*/Port 2222/' /etc/ssh/sshd_config`
4. Discovered service name is `ssh` not `sshd` on Ubuntu 24.04: `systemctl restart ssh`
5. Verified port 2222 listening, then removed temporary port 22 rule

**Lesson**: Always keep the Hostinger web console as a fallback. On Ubuntu 24.04, the service is `ssh.service`, not `sshd.service`.

### 2. Docker DNS Resolution Failure

**Problem**: `docker compose build` failed with `Temporary failure resolving 'deb.debian.org'`. The `apt-get update` inside the Dockerfile couldn't reach any Debian mirrors.

**Root Cause**: The host uses `systemd-resolved` with a stub resolver at `127.0.0.53`. Docker containers have their own network namespace and can't reach the host's loopback address.

**Solution**: Created `/etc/docker/daemon.json` with explicit DNS servers (Google 8.8.8.8 + Cloudflare 1.1.1.1) and restarted Docker. Build succeeded immediately after.

**Lesson**: Always configure explicit DNS for Docker on VPS instances that use systemd-resolved.

### 3. `src/data/` Excluded by .gitignore

**Problem**: Tests passed locally (449/449) but failed on VPS with `ModuleNotFoundError: No module named 'src.data'`.

**Root Cause**: `.gitignore` had `data/` which matched both the top-level `data/` directory (trade database, meant to be excluded) AND `src/data/` (source code, must be tracked).

**Solution**: Changed `data/` to `/data/` (leading slash = root-relative only). Committed the 7 missing `src/data/*.py` files, pushed, pulled on VPS.

**Lesson**: Always use `/` prefix in .gitignore for top-level exclusions to avoid accidentally excluding nested directories with the same name. Run `git ls-files` to verify what's actually tracked.

### 4. Ubuntu 24.04 Service Name

**Problem**: `systemctl restart sshd` returned `Unit sshd.service not found`.

**Root Cause**: Ubuntu 24.04 changed the SSH service name from `sshd` to `ssh`.

**Solution**: Use `systemctl restart ssh` instead.

### 5. Container Restart Loop (Expected)

**Problem**: After `docker compose up -d`, container kept restarting with exit code 0.

**Root Cause**: Not a bug — the trading agent checks if market is open, finds it closed (Friday evening), and exits cleanly. The `restart: unless-stopped` policy restarts it, it checks again, exits, repeats.

**Non-issue**: This is by design. When market opens (Monday 9:15 AM), the agent will start and run until 3:30 PM. The restart loop uses negligible resources (process starts and exits in <1 second).

---

## Daily Operations

### Start Paper Trading
```bash
ssh vps "cd /opt/stock-predict && docker compose up -d"
```

### Start Live Trading (Zerodha)
```bash
ssh vps "cd /opt/stock-predict && docker compose -f docker-compose.yml -f docker-compose.live.yml up -d"
```

### Check Status
```bash
ssh vps "docker ps"                          # Container status
ssh vps "docker logs -f stock-predict"       # Follow live logs
ssh vps "docker logs --tail 50 stock-predict" # Last 50 lines
```

### Deploy New Code
```bash
# On your Mac:
git add -A && git commit -m "description" && git push

# Then deploy:
ssh vps "cd /opt/stock-predict && git pull && docker compose build && docker compose up -d"
```

### Stop Trading
```bash
ssh vps "cd /opt/stock-predict && docker compose down"
```

### Emergency: Check VPS Health
```bash
ssh vps "free -h && df -h / && docker ps && ufw status"
```

### View Trade Database
```bash
ssh vps "sqlite3 /opt/stock-predict/data/trades.db '.tables'"
ssh vps "sqlite3 /opt/stock-predict/data/trades.db 'SELECT * FROM trades ORDER BY entry_time DESC LIMIT 10;'"
```

---

## Troubleshooting

### Can't SSH In
1. Check if VPS is running in Hostinger hPanel
2. Use Hostinger web terminal (serial console — always works)
3. Verify: `ufw status` shows port 2222 open
4. Verify: `ss -tlnp | grep ssh` shows sshd on 2222
5. Check fail2ban: `fail2ban-client status sshd` (your IP might be banned)
6. Unban yourself: `fail2ban-client set sshd unbanip YOUR_IP`

### Docker Build Fails
1. Check DNS: `docker run --rm alpine nslookup google.com`
2. If DNS fails, verify `/etc/docker/daemon.json` has explicit DNS
3. Restart Docker: `sudo systemctl restart docker`
4. Check disk space: `df -h /`

### Container Won't Start
1. Check logs: `docker logs stock-predict`
2. Check .env exists: `ls -la /opt/stock-predict/.env`
3. Test env vars: `docker compose run --rm --entrypoint python trading-agent -c "import os; print(os.getenv('OPENAI_API_KEY', 'MISSING'))"`

### Market Hours but No Trading
1. Check container is running: `docker ps`
2. Check it's not weekend/holiday
3. Check logs for errors: `docker logs --tail 100 stock-predict`
4. Verify broker credentials in .env

---

## Installed Software

| Package | Version | Purpose |
|---------|---------|---------|
| Docker Engine | 29.2.1 | Container runtime |
| Docker Compose | v2 (bundled) | Multi-container orchestration |
| UFW | default | Firewall management |
| fail2ban | 1.0.2 | SSH brute-force protection |
| unattended-upgrades | default | Automatic security patches |
| git | default | Code deployment |

---

## Security Checklist

- [x] Non-root user (`deploy`) for all operations
- [x] SSH key-only authentication (passwords disabled)
- [x] Non-standard SSH port (2222)
- [x] Root SSH login disabled
- [x] UFW firewall (only SSH port open)
- [x] Fail2ban (5 attempts = 1 hour ban)
- [x] Automatic security updates (reboot at 4 AM IST)
- [x] Kernel parameter hardening (SYN flood, ICMP, source routing)
- [x] Docker read-only filesystem
- [x] Docker capability drop (ALL)
- [x] Docker no-new-privileges
- [x] Non-root container user (`trading`)
- [x] .env file chmod 600
- [x] .env excluded from git
- [x] No secrets in Docker image
- [x] IST timezone enforced (host + container)
- [x] JSON logging with 10MB rotation (prevents disk fill)
- [x] 449 tests passing on VPS

---

*Last updated: February 13, 2026*
*Setup performed on: Hostinger KVM 4, Ubuntu 24.04 LTS*
