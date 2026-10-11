#!/bin/sh
# Install the headless browser stack on the server (run as root from /opt/aichain).
set -eu
cd /opt/aichain
.venv/bin/pip install -q -e '.[browser]'
getent group aichain-net >/dev/null || groupadd --system aichain-net
id aichain-egress >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin -g aichain-net aichain-egress
id aichain-browser >/dev/null 2>&1 || useradd --system --no-create-home --home-dir /var/lib/aichain-browser --shell /usr/sbin/nologin -U -G aichain-net aichain-browser
mkdir -p /opt/aichain-browser
PLAYWRIGHT_BROWSERS_PATH=/opt/aichain-browser/ms-playwright .venv/bin/playwright install --with-deps chromium
chmod -R a+rX /opt/aichain-browser
# Keep runtime data and secrets away from the browser users (they also get InaccessiblePaths).
chmod 700 /opt/aichain/data
chmod 600 /opt/aichain/.env
install -m 644 deploy/systemd/aichain-egress.service deploy/systemd/aichain-browser.service /etc/systemd/system/
systemctl daemon-reload
echo "installed; start with: systemctl enable --now aichain-egress aichain-browser"
