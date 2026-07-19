# Production image for the hosted demo (Hugging Face Spaces, Docker SDK).
# Serves the FastAPI app on port 7860 (the port HF Spaces expects).
FROM python:3.12-slim

# System deps: libpq for psycopg. Kept minimal.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install slim runtime deps first (better layer caching).
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# App code (only what the server needs at runtime).
COPY agent/ ./agent/
COPY server/ ./server/
COPY static/ ./static/

# Bind to the platform-provided $PORT (Render sets this); default 7860 locally.
ENV PORT=7860
EXPOSE 7860

# DATABASE_URL and ANTHROPIC_API_KEY are provided as host secrets at runtime.
# Shell form so ${PORT} is expanded at container start.
CMD uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-7860}
