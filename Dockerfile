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
 && rm -rf /var/lib/apt/lists/*
 
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

# run pipeline
CMD ["bash", "src/full.sh"]
