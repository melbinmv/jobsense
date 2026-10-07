FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and core tools first
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --upgrade setuptools==80.9.0 wheel

# Install mlflow first at a version that works with Python 3.12
RUN pip install --no-cache-dir "mlflow==2.17.2"

# Verify pkg_resources works before continuing
RUN python -c "import pkg_resources; print('pkg_resources OK')"

# Install remaining dependencies (with their transitive deps, e.g. h11)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sanity check: fail the build early if uvicorn's dependencies are missing
RUN python -c "import uvicorn, h11, httpx; print('uvicorn/h11/httpx OK')"

# Bake the embedding model into the image so cold starts don't download it.
# Placed before COPY src/ so this layer is cached across code changes.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY config.yaml .
COPY pyproject.toml .

# Install the project as a package
RUN pip install --no-cache-dir -e .

# Create data directories
RUN mkdir -p data/raw data/processed

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]