.PHONY: install models dev db up down logs clean

# Install Node dependencies for the API (outside Docker)
install:
	cd arcface-api && npm install

# Download ONNX models locally into arcface-api/models
models:
	mkdir -p arcface-api/models
	curl -L -o arcface-api/models/arcface.onnx https://huggingface.co/onnxmodelzoo/arcfaceresnet100-8/resolve/main/arcfaceresnet100-8.onnx
	curl -L -o arcface-api/models/scrfd.onnx https://huggingface.co/crj/dl-ws/resolve/main/scrfd_2.5g.onnx

# Launch only the Postgres service via Docker Compose
db:
	docker compose -f arcface-api/docker-compose.yml up -d db
	@export DATABASE_URL=postgres://postgres:postgres@localhost:5432/face_db

# Run the API locally with ts-node-dev (requires DATABASE_URL env)
run: install db
	cd arcface-api && npm run dev

# Build and start the full stack (API + Postgres) with Docker Compose
up:
	docker compose -f arcface-api/docker-compose.yml up --build

# Stop and remove running containers
down:
	docker compose -f arcface-api/docker-compose.yml down

# Tail logs from running containers
logs:
	docker compose -f arcface-api/docker-compose.yml logs -f

# Remove containers, networks, and volumes
clean:
	docker compose -f arcface-api/docker-compose.yml down -v

# Clean and rebuild everything from scratch
reset: clean
	docker compose -f arcface-api/docker-compose.yml up --build

