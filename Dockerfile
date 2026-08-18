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
