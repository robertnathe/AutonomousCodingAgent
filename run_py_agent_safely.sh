#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="coding-agent-sandbox"
SCRIPT_NAME="AdvancedCodingAgent.py"

echo "Building sandbox image..."
cat > Dockerfile << EOF
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl gcc && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 -s /bin/bash agentuser
WORKDIR /app
COPY --chown=agentuser:agentuser . /app
USER agentuser
RUN pip install --no-cache-dir --user groq chromadb pyyaml
ENV PATH="/home/agentuser/.local/bin:\${PATH}"
CMD ["python", "$SCRIPT_NAME"]
EOF

docker build -t $IMAGE_NAME .

echo -e "\nStarting sandboxed agent...\n"
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
  $IMAGE_NAME
  
