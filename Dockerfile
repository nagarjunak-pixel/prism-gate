FROM node:22-slim AS ui-builder
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PRISM_UI_DIST_DIR=/app/ui/dist

WORKDIR /app
RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN uv pip install --system --no-cache .

COPY --from=ui-builder /ui/dist /app/ui/dist

EXPOSE 8000
CMD ["prism", "serve", "--host", "0.0.0.0", "--port", "8000"]
