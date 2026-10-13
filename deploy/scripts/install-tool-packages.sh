#!/bin/sh
# Install the tool package allowlist (deploy/tool-packages.txt) for the system python3 into a
# root-owned directory that SystemdExecutor appends to sys.path inside the sandbox. Run as root
# from /opt/aichain once, and again after changing deploy/tool-packages.txt.
#
#   sh deploy/scripts/install-tool-packages.sh [/opt/aichain-tool-packages]
#
# Nothing else from the system's Python is visible to tools (python3 -I -S, dist-packages are
# InaccessiblePaths); this directory is read-only for them like the rest of the system.
set -eu
DIR=${1:-/opt/aichain-tool-packages}
PY=/usr/bin/python3
REQ=$(dirname "$0")/../tool-packages.txt
umask 022
rm -rf "$DIR.new"
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" --target "$DIR.new" --only-binary :all: -r "$REQ"
elif "$PY" -m pip --version >/dev/null 2>&1; then
    "$PY" -m pip install --no-cache-dir --disable-pip-version-check --target "$DIR.new" \
        --only-binary :all: -r "$REQ"
else
    echo "need uv or pip for $PY (e.g. apt install python3-pip)" >&2
    exit 1
fi
rm -rf "$DIR.new/bin"  # console scripts are not used in the sandbox
# Precompile (paths as they will be after the move): the sandbox cannot write __pycache__, and
# compiling pandas on every call would eat most of a tool's time limit.
"$PY" -m compileall -q -f -s "$DIR.new" -p "$DIR" "$DIR.new" >/dev/null || true
chown -R root:root "$DIR.new"
chmod -R u=rwX,go=rX "$DIR.new"
# Check with the same interpreter flags the sandbox uses.
"$PY" -I -S -c "import sys; sys.path.append('$DIR.new')
import jsonschema, numpy, pandas, sympy, bs4, yaml, dateutil, regex
print('ok: numpy', numpy.__version__, 'pandas', pandas.__version__)"
rm -rf "$DIR.old"
if [ -d "$DIR" ]; then mv "$DIR" "$DIR.old"; fi
mv "$DIR.new" "$DIR"
rm -rf "$DIR.old"
echo "installed into $DIR; verify with: .venv/bin/python deploy/scripts/executor-escape-tests.py"
