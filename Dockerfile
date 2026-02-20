# Use a Python image with uv pre-installed
ARG PYTHON_VERSION=3.12
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-trixie-slim

WORKDIR /app

# System deps (OpenCV / audio deps commonly needed)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Copy only dependency files first for better layer caching
COPY pyproject.toml ./
# If you later decide to generate uv.lock, this will get picked up automatically
COPY uv.lock* ./

# Install deps (no lock required; if uv.lock exists, uv will use it)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev

# Now copy the rest of the app
COPY . .

EXPOSE 7860

# Place the virtual environment in PATH to avoid needing uv run
ENV PATH="/app/.venv/bin:$PATH"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
