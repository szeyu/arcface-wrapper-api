FROM node:24-slim AS build

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*

COPY package*.json ./

# Use npm ci for faster, reproducible installs with cache mount
RUN --mount=type=cache,target=/root/.npm \
    npm ci

COPY tsconfig.json ./
COPY src ./src

# Copy models directory from host if it exists (for faster builds)
# This allows using local models to skip download entirely
COPY models ./models/

# Download ONNX models with cache mount for persistence across builds
# Priority: 1) Use models from host (if copied), 2) Use Docker cache, 3) Download
# Note: If models directory is empty or missing files, they will be downloaded/cached
RUN --mount=type=cache,target=/tmp/models-cache \
    mkdir -p models && \
    if [ ! -f models/face_recognition.onnx ]; then \
      if [ -f /tmp/models-cache/face_recognition.onnx ]; then \
        echo "✓ Using cached face_recognition.onnx from Docker cache" && \
        cp /tmp/models-cache/face_recognition.onnx models/face_recognition.onnx; \
      else \
        echo "↓ Downloading face_recognition.onnx (this may take a while)..." && \
        curl -fL -o models/face_recognition.onnx https://huggingface.co/deepghs/insightface/resolve/main/buffalo_l/w600k_r50.onnx && \
        cp models/face_recognition.onnx /tmp/models-cache/face_recognition.onnx || true && \
        echo "✓ face_recognition.onnx downloaded and cached"; \
      fi; \
    else \
      echo "✓ Using face_recognition.onnx from host"; \
    fi && \
    if [ ! -f models/retinaface_resnet50.onnx ]; then \
      if [ -f /tmp/models-cache/retinaface_resnet50.onnx ]; then \
        echo "✓ Using cached retinaface_resnet50.onnx from Docker cache" && \
        cp /tmp/models-cache/retinaface_resnet50.onnx models/retinaface_resnet50.onnx; \
      else \
        echo "↓ Downloading retinaface_resnet50.onnx (this may take a while)..." && \
        curl -fL -o models/retinaface_resnet50.onnx https://storage.googleapis.com/ailia-models/retinaface/retinaface_resnet50.onnx && \
        cp models/retinaface_resnet50.onnx /tmp/models-cache/retinaface_resnet50.onnx || true && \
        echo "✓ retinaface_resnet50.onnx downloaded and cached"; \
      fi; \
    else \
      echo "✓ Using retinaface_resnet50.onnx from host"; \
    fi

RUN npm run build

FROM node:24-slim

WORKDIR /app

COPY --from=build /app/dist ./dist
COPY --from=build /app/models ./models
COPY package*.json ./

# Use npm ci --only=production with cache mount for faster installs
RUN --mount=type=cache,target=/root/.npm \
    npm ci --only=production

EXPOSE 3000

CMD ["node", "dist/server.js"]
