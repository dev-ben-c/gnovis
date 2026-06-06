#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Gnovis Deploy ==="

# 1. Python venv
if [ ! -d "venv" ]; then
    echo "Creating Python venv..."
    python3 -m venv venv
fi
source venv/bin/activate
echo "Python: $(python --version)"

# 2. Install package
echo "Installing gnovis..."
pip install -q -e ".[vectors,dev]"

# 3. Ollama embedding model
if command -v ollama &>/dev/null; then
    if ! ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
        echo "Pulling nomic-embed-text model..."
        ollama pull nomic-embed-text
    else
        echo "nomic-embed-text already available"
    fi
else
    echo "WARNING: Ollama not found. Vector search will be disabled."
    echo "  Install from https://ollama.ai then run: ollama pull nomic-embed-text"
fi

# 4. Run tests
echo ""
echo "Running tests..."
pytest tests/ -q

# 5. Backfill embeddings if DB exists
DB_PATH="${ENGRAM_DB:-$HOME/.engram/memory.db}"
if [ -f "$DB_PATH" ]; then
    echo ""
    echo "Backfilling embeddings for any new memories..."
    python -c "
from engram.store import MemoryStore
s = MemoryStore('$DB_PATH')
result = s.backfill_embeddings()
if result['total'] > 0:
    print(f'  Embedded {result[\"embedded\"]}/{result[\"total\"]} memories')
else:
    print('  All memories already have embeddings')
s.close()
"
fi

# 6. Check for systemd service
echo ""
if systemctl --user is-active engram-mcp &>/dev/null; then
    echo "Restarting engram-mcp user service..."
    systemctl --user restart engram-mcp
    echo "Service restarted."
elif sudo systemctl is-active engram-mcp &>/dev/null; then
    echo "Restarting engram-mcp system service..."
    sudo systemctl restart engram-mcp
    echo "Service restarted."
else
    # Check for running SSE process
    SSE_PID=$(pgrep -f "engram.server.*--transport sse" || true)
    if [ -n "$SSE_PID" ]; then
        echo "Restarting SSE server (PID $SSE_PID)..."
        kill "$SSE_PID"
        sleep 1
        nohup "$SCRIPT_DIR/venv/bin/python" -m engram.server --transport sse --host 0.0.0.0 --port 8093 > /tmp/engram-sse.log 2>&1 &
        echo "SSE server restarted (PID $!)"
        sleep 2
        if curl -sf http://localhost:8093/health > /dev/null; then
            echo "Health check: OK"
        else
            echo "WARNING: Health check failed. Check /tmp/engram-sse.log"
        fi
    else
        echo "No running Gnovis service found."
        echo "  Start with: python -m engram.server --transport sse --host 0.0.0.0 --port 8093"
        echo "  Or add to Claude Code: claude mcp add engram -- $SCRIPT_DIR/venv/bin/python -m engram.server"
    fi
fi

echo ""
echo "=== Deploy complete ==="
