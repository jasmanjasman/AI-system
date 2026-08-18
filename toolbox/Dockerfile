# Dockerfile
FROM python:3.12-slim

# Prevents .pyc files + unbuffered logs (so you see output immediately in docker logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements FIRST and install — this layer caches unless requirements change,
# so code edits don't trigger a full reinstall. Classic Docker layer-caching trick.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the app code (changes often — kept in its own late layer)
COPY app ./app

# Run as an unprivileged user rather than root
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# No --reload in the image default; compose overrides this for dev (see below)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
