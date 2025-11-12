# ArcFace Face Vector API

A Node.js API for face recognition using ArcFace embeddings and SCRFD face detection.

## Requirements

- Node.js 20+
- Docker + Docker Compose
- PostgreSQL (via Docker)

## Quick Start

### Local Development

```bash
# Install dependencies
make install

# Download ONNX models
make models

# Make scripts executable (if you plan to use them)
make chmod-scripts

# Start PostgreSQL database
make db

# Run the API locally
make run
```

### Docker Compose

```bash
# Build and start everything
make up

# Or with Docker Compose directly
docker compose -f arcface-api/docker-compose.yml up --build
```

- API runs on `http://localhost:3000/api`
- Postgres runs on `localhost:5432` (DB: face_db)
- ONNX models (ArcFace + SCRFD) are downloaded automatically during Docker build

## Makefile Commands

```bash
make install        # Install Node dependencies
make models         # Download ONNX models
make chmod-scripts  # Make scripts executable
make db             # Start PostgreSQL only (detached)
make run            # Run API locally (requires db to be running)
make up             # Build and start full stack with Docker
make down           # Stop containers
make logs           # View container logs
make clean          # Remove containers and volumes
make reset          # Clean and rebuild everything
```

## API Endpoints

### Store Face Embedding
Store a face embedding from an image file. Requires SCRFD to detect at least one face.

**POST** `/api/store_embedding`

```bash
curl -X POST http://localhost:3000/api/store_embedding \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/absolute/path/to/image.jpg"}'
```

**Response:**
```json
{"id": "550e8400-e29b-41d4-a716-446655440000"}
```

### Compare Two Faces
Compare two face images and get similarity metrics.

**POST** `/api/compare`

```bash
curl -X POST http://localhost:3000/api/compare \
  -H "Content-Type: application/json" \
  -d '{"image_path_A": "/path/to/image1.jpg", "image_path_B": "/path/to/image2.jpg"}'
```

**Response:**
```json
{
  "cosine": 0.95,
  "euclidean": 2.34
}
```

### Search Similar Faces
Find the most similar faces in the database.

**POST** `/api/search`

```bash
curl -X POST http://localhost:3000/api/search \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/path/to/query.jpg", "top_k": 5}'
```

**Response:**
```json
[
  {"id": "550e8400-e29b-41d4-a716-446655440000", "cosine": 0.98},
  {"id": "660e8400-e29b-41d4-a716-446655440001", "cosine": 0.92}
]
```

### Get Image by ID
Retrieve an image by its UUID. Also saves the image to `arcface-api/output/{id}.{ext}`.

**GET** `/api/image/:id`

```bash
curl -X GET http://localhost:3000/api/image/550e8400-e29b-41d4-a716-446655440000
```

**Response:**
```json
{
  "image_base64": "iVBORw0KGgo...",
  "saved_to": "/path/to/arcface-api/output/550e8400-e29b-41d4-a716-446655440000.jpeg"
}
```

### List Stored Embeddings
Get a list of stored embeddings.

**GET** `/api/list?limit=N`

```bash
curl -X GET "http://localhost:3000/api/list?limit=10"
```

**Response:**
```json
[
  {"id": "550e8400-e29b-41d4-a716-446655440000", "created_at": "2024-01-01T12:00:00Z"},
  {"id": "660e8400-e29b-41d4-a716-446655440001", "created_at": "2024-01-01T11:00:00Z"}
]
```

### Delete Embedding
Delete an embedding by ID.

**DELETE** `/api/item/:id`

```bash
curl -X DELETE http://localhost:3000/api/item/550e8400-e29b-41d4-a716-446655440000
```

**Response:**
```json
{"deleted_id": "550e8400-e29b-41d4-a716-446655440000"}
```

## Using Scripts

Convenience scripts are available in the `scripts/` folder. **Remember to run `make chmod-scripts` first** to make the scripts executable:

```bash
# Make scripts executable (run this first!)
make chmod-scripts

# Store an embedding
./scripts/store-embedding.sh examples/face1.jpeg

# Compare two faces
./scripts/compare.sh examples/face1.jpeg examples/face2.jpeg

# Search for similar faces
./scripts/search.sh examples/face1.jpeg 5

# Get an image by ID
./scripts/get-image.sh <uuid>

# List stored embeddings
./scripts/list.sh 10

# Delete an embedding
./scripts/delete.sh <uuid>
```

All scripts accept an optional API URL as the last parameter:
```bash
./scripts/store-embedding.sh examples/face1.jpeg http://localhost:3000
```

## Example Workflow

```bash
# 1. Store first face
./scripts/store-embedding.sh examples/face1.jpeg
# Response: {"id": "abc-123-def"}

# 2. Store second face
./scripts/store-embedding.sh examples/face2.jpeg
# Response: {"id": "xyz-456-uvw"}

# 3. Compare them
./scripts/compare.sh examples/face1.jpeg examples/face2.jpeg

# 4. Search for similar faces
./scripts/search.sh examples/face1.jpeg 3

# 5. List all stored faces
./scripts/list.sh

# 6. Get and save an image
./scripts/get-image.sh abc-123-def
# Image saved to: arcface-api/output/abc-123-def.jpeg
```

## Output Folder

When you call `GET /api/image/:id`, the image is automatically saved to the `arcface-api/output/` folder with the UUID as the filename (e.g., `a69e09ac-e733-4c4e-8ea4-657e6a54d741.jpeg`). The image format (jpeg/png/gif/webp) is automatically detected from the stored image data.

## Testing Face Detection

To verify that face detection is working correctly, you can test with the `examples/box.jpeg` image, which does not contain a face:

```bash
# This should fail with "no_face_detected" error
./scripts/store-embedding.sh examples/box.jpeg
```

**Expected Response:**
```json
{"error": "no_face_detected"}
```

This confirms that the SCRFD face detector correctly rejects images without faces.

## Notes

- **Face Detection**: All endpoints require SCRFD to detect at least one face in the image. If no face is detected, the request fails with `{"error": "no_face_detected"}`.
- **Image Paths**: Use absolute paths or paths relative to where the API process is running (typically `arcface-api/` directory).
- **Image Formats**: Supports JPEG, PNG, GIF, and WebP formats.
- **Example Images**: The `examples/` folder contains sample face images (`face1.jpeg`, `face2.jpeg`, etc.) for testing, and `box.jpeg` to test face detection rejection.
