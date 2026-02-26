FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git openssh-client \
    libcairo2 \
    libpango-1.0-0 libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi8 libglib2.0-0 \
    shared-mime-info \
    fonts-dejavu fonts-liberation \
    pandoc \
    nodejs npm \
 && rm -rf /var/lib/apt/lists/* \
 && npm install -g @github/copilot


# GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends gh \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

# Select pipeline: agentbased (default) or chatbased
ARG PIPELINE=agentbased
ENV PIPELINE=${PIPELINE}
CMD if [ "${PIPELINE}" = "agentbased" ]; then bash agentic/full.sh; \
    elif [ "${PIPELINE}" = "chatbased" ]; then bash src/full.sh; \
    else echo "[err] Unknown PIPELINE value: ${PIPELINE}. Use 'agentbased' or 'chatbased'."; exit 1; \
    fi