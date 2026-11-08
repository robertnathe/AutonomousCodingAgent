# bash ./run_cpp_safely_agent.sh --rebuild
#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-coding-agent-sandbox}"
BINARY_NAME="${BINARY_NAME:-output}"
MEMORY_LIMIT="${MEMORY_LIMIT:-4g}"
CPU_LIMIT="${CPU_LIMIT:-2}"
PIDS_LIMIT="${PIDS_LIMIT:-256}"
TMP_SIZE="${TMP_SIZE:-1g}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}→${NC} $*"; }
ok()    { echo -e "${GREEN}✓${NC} $*"; }
warn()  { echo -e "${YELLOW}!${NC} $*"; }
die()   { echo -e "${RED}✗${NC} $*" >&2; exit 1; }

REBUILD=false
PASS_THROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild|-r) REBUILD=true; shift ;;
    --help|-h)
      cat <<EOF
Usage: $(basename "$0") [OPTIONS] [-- agent-args...]

Safely build and run Advanced_Coding_Agent_05 (and related sources)
inside a hardened Docker sandbox.

Options:
  -r, --rebuild     Force rebuild of the sandbox image
  -h, --help        Show this help

Environment variables:
  GROQ_API_KEY / OPENROUTER_API_KEY / GOOGLE_API_KEY
                    At least one is required for LLM calls
  MEMORY_LIMIT      Docker memory limit (default: 4g)
  CPU_LIMIT         Docker CPU limit (default: 2)
  PIDS_LIMIT        Max processes (default: 256)
  TMP_SIZE          Size of the executable /tmp tmpfs (default: 1g)
  BINARY_NAME       Binary produced by make (default: output)
  IMAGE_NAME        Docker image tag (default: coding-agent-sandbox)

Any arguments after -- are passed to the agent binary.
EOF
      exit 0
      ;;
    --) shift; PASS_THROUGH=("$@"); break ;;
    *)  PASS_THROUGH+=("$1"); shift ;;
  esac
done

command -v docker >/dev/null 2>&1 || die "Docker is not installed or not in PATH"
docker info >/dev/null 2>&1 || die "Docker daemon is not running (or you lack permission)"

if [[ ! -f Makefile ]]; then
  warn "No Makefile found – will try a direct g++ fallback for Advanced_Coding_Agent_00.cpp"
fi

if ! ls *.cpp >/dev/null 2>&1; then
  warn "No .cpp files found in the current directory"
fi

if [[ -z "${GROQ_API_KEY:-}" && -z "${OPENROUTER_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
  warn "No LLM API keys set (GROQ_API_KEY / OPENROUTER_API_KEY / GOOGLE_API_KEY)"
  warn "The agent will fail to talk to any backend"
fi

# IMPORTANT: /tmp must be executable – the agent compiles to /tmp/agent_*.out
DOCKERFILE_CONTENT=$(cat <<'EOF'
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        make \
        libcurl4-openssl-dev \
        libssl-dev \
        libyaml-cpp-dev \
        nlohmann-json3-dev \
        ca-certificates \
        curl \
        python3 \
        python3-pip \
        python3-venv \
        findutils \
        coreutils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Ubuntu 24.04 already has a UID-1000 user ("ubuntu"). Rename for clarity.
RUN usermod -l agentuser ubuntu \
 && groupmod -n agentuser ubuntu \
 && usermod -d /home/agentuser -m agentuser

WORKDIR /app
USER agentuser

CMD ["./output"]
EOF
)

NEED_BUILD=false
if $REBUILD || ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  NEED_BUILD=true
else
  CURRENT_HASH=$(echo "$DOCKERFILE_CONTENT" | sha256sum | cut -d' ' -f1)
  CACHED_HASH=$(docker image inspect \
    --format '{{index .Config.Labels "sandbox.dockerfile.hash"}}' \
    "$IMAGE_NAME" 2>/dev/null || echo "")
  if [[ "$CURRENT_HASH" != "$CACHED_HASH" ]]; then
    NEED_BUILD=true
  fi
fi

if $NEED_BUILD; then
  info "Building sandbox image ($IMAGE_NAME)..."
  DOCKER_BUILDKIT=1 docker build \
    --label "sandbox.dockerfile.hash=$(echo "$DOCKERFILE_CONTENT" | sha256sum | cut -d' ' -f1)" \
    -t "$IMAGE_NAME" \
    - <<< "$DOCKERFILE_CONTENT"
  ok "Image built"
else
  ok "Using existing image $IMAGE_NAME (pass --rebuild to force)"
fi

info "Starting sandboxed C++ coding agent"
echo "  • Only the current directory is mounted at /app"
echo "  • Runs as non-root (UID 1000)"
echo "  • Memory ≤ $MEMORY_LIMIT  |  CPUs ≤ $CPU_LIMIT  |  PIDs ≤ $PIDS_LIMIT"
echo "  • /tmp is a $TMP_SIZE executable tmpfs (required for compile_cpp)"
echo "  • Capabilities dropped + no-new-privileges"
echo "  Press Ctrl+C to stop"
echo

cleanup() {
  docker rm -f coding-agent-run 2>/dev/null || true
}
trap cleanup EXIT

ENV_ARGS=()
for key in GROQ_API_KEY OPENROUTER_API_KEY GOOGLE_API_KEY; do
  if [[ -n "${!key:-}" ]]; then
    ENV_ARGS+=(-e "$key")
  fi
done

docker run --rm -it \
  --name coding-agent-run \
  -v "$(pwd)":/app \
  -u 1000:1000 \
  --memory="$MEMORY_LIMIT" \
  --cpus="$CPU_LIMIT" \
  --pids-limit="$PIDS_LIMIT" \
  --cap-drop=ALL \
  --security-opt=no-new-privileges:true \
  --tmpfs "/tmp:rw,exec,nosuid,size=${TMP_SIZE}" \
  --ulimit nofile=1024:1024 \
  "${ENV_ARGS[@]}" \
  -e HOME=/home/agentuser \
  -e TERM=xterm-256color \
  "$IMAGE_NAME" \
  bash -c '
    set -euo pipefail
    echo "[sandbox] Building agent..."

    if [[ -f Makefile ]]; then
      make clean 2>/dev/null || true
      make
    else
      # Fallback when no Makefile is present
      SRC="Advanced_Coding_Agent_00.cpp"
      if [[ ! -f "$SRC" ]]; then
        echo "No Makefile and $SRC not found" >&2
        exit 1
      fi
      g++ -std=c++20 -O2 -Wall \
          "$SRC" -o "'"$BINARY_NAME"'" \
          -lcurl -lssl -lcrypto -pthread -lyaml-cpp
    fi

    if [[ ! -x ./'"$BINARY_NAME"' ]]; then
      echo "Binary ./'"$BINARY_NAME"' was not produced" >&2
      exit 1
    fi

    echo "[sandbox] Launching agent..."
    exec ./'"$BINARY_NAME"' '"${PASS_THROUGH[*]@Q}"'
  '
