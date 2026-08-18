#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="coding-agent-sandbox"
BINARY_NAME="output"

echo "Building sandbox image..."

cat > Dockerfile << 'EOF'
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    make \
    libcurl4-openssl-dev \
    libyaml-cpp-dev \
    libssl-dev \
    nlohmann-json3-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 already has a user with UID 1000 ("ubuntu").
# We just rename it for clarity and make sure the home directory is correct.
RUN usermod -l agentuser ubuntu && \
    groupmod -n agentuser ubuntu && \
    usermod -d /home/agentuser -m agentuser

WORKDIR /app
USER agentuser

CMD ["./output"]
EOF

docker build -t "$IMAGE_NAME" .

echo -e "\nStarting sandboxed C++ agent...\n"
echo "→ Only the current directory is visible inside"
echo "→ Runs as non-root user (UID 1000)"
echo "→ CPU/memory limited"
echo -e "Press Ctrl+C to stop\n"

docker run --rm -it \
  --name coding-agent-run \
  -v "$(pwd)":/app \
  -u 1000:1000 \
  --memory=3g \
  --cpus=2 \
  -e GROQ_API_KEY \
  "$IMAGE_NAME" \
  bash -c "make clean 2>/dev/null || true; make && ./$BINARY_NAME"