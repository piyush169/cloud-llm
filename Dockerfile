FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

WORKDIR /app

# System deps (optional but useful for many ML wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app code + already-built indexes (if present in repo)
COPY . .

EXPOSE 8501

# Only run Streamlit app (no chunking/indexing commands)
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
