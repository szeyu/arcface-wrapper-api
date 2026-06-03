-include .env
export

.PHONY: install install-dev models dev db up-db up-rustfs up down logs clean chmod-scripts reset run lint lint-fix test test-storage test-integration test-management test-examples verify-recognition-examples benchmark-performance

COMPOSE := docker compose --env-file .env -f docker-compose.yml

# Install production Node dependencies only (for Docker/production)
install:
	npm ci --omit=dev

# Install all Node dependencies including dev (for local development)
install-dev:
	npm install

# Lint TypeScript codebase (check only)
lint: install-dev
	npm run lint

# Lint and auto-fix TypeScript codebase
lint-fix: install-dev
	npm run lint:fix

# Run RustFS storage integration test only
test-storage: install-dev up-rustfs
	S3_ENDPOINT=http://localhost:9000 npm test -- src/__tests__/storage.test.ts

# Run full test suite
test: install-dev models up-db up-rustfs
	S3_ENDPOINT=http://localhost:9000 npm test

# Run integration test only
test-integration: install-dev models up-db up-rustfs
	S3_ENDPOINT=http://localhost:9000 npm test -- src/__tests__/integration.test.ts

# Run management test only
test-management: install-dev models up-db up-rustfs
	S3_ENDPOINT=http://localhost:9000 npm test -- src/__tests__/management.test.ts

# Run examples fixture coverage test only
test-examples: install-dev models up-db up-rustfs
	S3_ENDPOINT=http://localhost:9000 npm test -- src/__tests__/examples.test.ts

# Run positive and negative recognition examples against a running API
verify-recognition-examples:
	./scripts/verify-recognition-examples.sh

# Run API benchmark against a running API and write reports to benchmarks/results/
benchmark-performance: install-dev
	npm run benchmark

# Download ONNX models locally into models/ (only if they don't exist)
models:
	@mkdir -p models
	@if [ ! -f models/face_recognition.onnx ]; then \
		echo "↓ Downloading face_recognition.onnx..."; \
		curl -fL -o models/face_recognition.onnx https://huggingface.co/deepghs/insightface/resolve/main/buffalo_l/w600k_r50.onnx; \
		echo "✓ face_recognition.onnx downloaded"; \
	else \
		echo "✓ face_recognition.onnx already exists, skipping download"; \
	fi
	@if [ ! -f models/retinaface_resnet50.onnx ]; then \
		echo "↓ Downloading retinaface_resnet50.onnx..."; \
		curl -fL -o models/retinaface_resnet50.onnx https://storage.googleapis.com/ailia-models/retinaface/retinaface_resnet50.onnx; \
		echo "✓ retinaface_resnet50.onnx downloaded"; \
	else \
		echo "✓ retinaface_resnet50.onnx already exists, skipping download"; \
	fi
	@echo "✓ All models ready"


# Launch only the Postgres service via Docker Compose
up-db:
	$(COMPOSE) up -d db

# Launch only the RustFS service via Docker Compose
up-rustfs:
	$(COMPOSE) up -d rustfs

# Run the API locally (requires db and RustFS running)
run: install-dev models up-db up-rustfs
	npm run dev

# Build and start the full stack (API + Postgres) with Docker Compose
up:
	$(COMPOSE) up --build

# Stop and remove running containers
down:
	$(COMPOSE) down

# Tail logs from running containers
logs:
	$(COMPOSE) logs -f

# Remove containers, networks, and volumes
clean:
	$(COMPOSE) down -v

# Clean and rebuild everything from scratch
reset: clean
	$(COMPOSE) up --build

# Make all scripts executable
chmod-scripts:
	chmod +x scripts/*
