# FaceVector Engine

A production-ready face recognition and vector similarity search engine. Built with Node.js/TypeScript, this API provides a two-step workflow for face detection, customer enrollment, and recognition using InsightFace Buffalo-L embeddings, RetinaFace detection, and PostgreSQL pgvector for efficient similarity search.

> 📚 **For detailed technical documentation**, see [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md) - comprehensive guide covering image processing pipelines, coordinate transformations, model inference, and performance optimizations.

## Features

- ✅ **Face Detection** - Detect faces with 5 facial landmarks using RetinaFace ResNet50
- ✅ **Face Enrollment** - Enroll customers with face embeddings for recognition
- ✅ **Face Recognition** - Identify customers by searching similar face embeddings
- ✅ **Multipart Upload** - Upload images as files (PNG, JPG, WEBP, AVIF) with automatic normalization and scaling
- ✅ **Two-Step Workflow** - Detect faces first, then enroll or recognize them
- ✅ **Vector Similarity Search** - PostgreSQL pgvector with optimized ivfflat indexing
- ✅ **512-dim Embeddings** - InsightFace Buffalo-L embeddings for high-accuracy face recognition
- ✅ **Docker Support** - Easy deployment with Docker Compose
- ✅ **TypeScript** - Fully typed codebase

## Architecture Overview

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
        USER[User/Application]
    end

    subgraph API["API Server (Express + TypeScript)"]
        DETECT["`POST /faces/detect
Detect & Store Faces`"]
        ENROLL["`POST /faces/enroll
Enroll Customer`"]
        RECOGNIZE["`POST /faces/recognize
Recognize Customer`"]
        GET["`GET /faces/:face_id
Get Face Image`"]
    end

    subgraph Models["AI Models (ONNX Runtime)"]
        RETINAFACE["`RetinaFace ResNet50
Face Detection`"]
        RECOGNITION["`InsightFace Buffalo-L
512-dim Embeddings`"]
    end

    subgraph Storage["Storage Layer"]
        DETECTED[("`detected_faces table
Face metadata`")]
        ENROLLED[("`enrolled_customers table
Vector embeddings`")]
        RUSTFS["`RustFS S3
Object Storage`"]
    end

    USER -->|Upload Image| DETECT
    DETECT -->|Detect Faces| RETINAFACE
    RETINAFACE -->|Bounding Boxes| DETECT
    DETECT -->|Store Metadata| DETECTED
    DETECT -->|Store Images| RUSTFS
    DETECT -->|face_id| USER

    USER -->|face_id + customer_info| ENROLL
    ENROLL -->|Read Face| RUSTFS
    ENROLL -->|Generate Embedding| RECOGNITION
    RECOGNITION -->|512-dim Vector| ENROLL
    ENROLL -->|Store| ENROLLED

    USER -->|face_id| RECOGNIZE
    RECOGNIZE -->|Read Face| RUSTFS
    RECOGNIZE -->|Generate Embedding| RECOGNITION
    RECOGNIZE -->|Vector Search| ENROLLED
    ENROLLED -->|Matched Customers| RECOGNIZE
    RECOGNIZE -->|Results| USER

    USER -->|face_id| GET
    GET -->|Read| RUSTFS
    RUSTFS -->|Image| USER

    style RETINAFACE fill:#e1f5ff
    style RECOGNITION fill:#e1f5ff
    style DETECTED fill:#fff4e1
    style ENROLLED fill:#fff4e1
    style RUSTFS fill:#ffe1f5
```

## Requirements

- Node.js 24+
- Docker + Docker Compose
- PostgreSQL with pgvector extension

## Quick Start

### Local Development

```bash
# Install dependencies
make install

# Download ONNX models
make models

# Make scripts executable
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
docker compose up --build
```

- API runs on `http://localhost:3000/api`
- Postgres runs on `localhost:5432` (DB: face_db)

## API Endpoints

### 1. Detect Faces

Detect all faces in an uploaded image, store face crops and metadata, and return face IDs.

**POST** `/api/faces/detect`

**Request:** Multipart form-data
- `file` (required): Image file (PNG, JPG, WEBP, AVIF, max 10MB)
- `identifier` (optional): Client-provided identifier for tracking

```bash
# Using curl
curl -X POST http://localhost:3000/api/faces/detect \
  -F "file=@examples/elon_musk_enroll.jpg" \
  -F "identifier=customer_123"

# Using script
./scripts/faces-detect.sh examples/elon_musk_enroll.jpg customer_123
```

**Response:**
```json
[
  {
    "face_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "position": {
      "x": 450,
      "y": 300,
      "width": 600,
      "height": 850
    },
    "confidence": 0.998,
    "file_name": "a1b2c3d4-e5f6-7890-abcd-ef1234567890.jpg"
  }
]
```

### 2. Get Face Image

Retrieve a stored face image by its face_id.

**GET** `/api/faces/:face_id`

```bash
# Using curl
curl -X GET http://localhost:3000/api/faces/a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
  -o face.jpg

# Using script (saves to output/)
./scripts/faces-get-image.sh a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

**Response:** Binary JPEG image

### 3. Enroll Customer

Enroll a customer by associating a detected face with customer information and generating an embedding.

**POST** `/api/faces/enroll`

**Request:** JSON
- `face_id` (required): UUID from `/faces/detect`
- `customer_identifier` (required): Unique customer identifier
- `customer_name` (optional): Customer's name
- `customer_metadata` (optional): Additional metadata as JSON

```bash
# Using curl
curl -X POST http://localhost:3000/api/faces/enroll \
  -H "Content-Type: application/json" \
  -d '{
    "face_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "customer_identifier": "CUST001",
    "customer_name": "John Doe",
    "customer_metadata": {"age": 30, "membership": "gold"}
  }'

# Using script
./scripts/faces-enroll.sh a1b2c3d4-e5f6-7890-abcd-ef1234567890 CUST001 "John Doe"
```

**Response:**
```json
{
  "customer_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
  "customer_identifier": "CUST001",
  "customer_name": "John Doe",
  "created_at": "2024-12-02T10:30:00.000Z"
}
```

### 4. Recognize Face

Recognize a customer by searching for similar enrolled face embeddings.

**POST** `/api/faces/recognize`

**Request:** JSON
- `face_id` (required): UUID from `/faces/detect`
- `min_confidence` (optional): Minimum similarity required to return a match. Defaults to `FACE_RECOGNITION_CONFIDENCE_THRESHOLD`.

`confidence_score` is cosine similarity, not a calibrated probability. The default threshold of `0.5` is intended to avoid false positives on normal probes. If a real same-person image falls below this value, prefer adding stronger or multiple enrollment images for that person before lowering the global threshold.

```bash
# Using curl
curl -X POST http://localhost:3000/api/faces/recognize \
  -H "Content-Type: application/json" \
  -d '{"face_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890", "min_confidence": 0.5}'

# Using script
./scripts/faces-recognize.sh a1b2c3d4-e5f6-7890-abcd-ef1234567890 http://localhost:3000 0.5
```

**Response:**
```json
[
  {
    "customer_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
    "customer_identifier": "CUST001",
    "customer_name": "John Doe",
    "confidence_score": 0.6708
  },
  {
    "customer_id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
    "customer_identifier": "CUST002",
    "customer_name": "Jane Smith",
    "confidence_score": 0.4123
  }
]
```

Results are sorted by confidence score (descending), returns up to 10 matches that pass the configured confidence threshold. An empty array (`[]`) means the nearest enrolled face did not pass the threshold and should be treated as "no match".

## Management Endpoints

Administrative endpoints for listing and managing detected faces and enrolled customers.

### List Detected Faces

**GET** `/api/management/faces?limit=50&offset=0`

Returns paginated list of all detected faces with metadata.

```bash
./scripts/management-list-faces.sh
```

### List Enrolled Customers

**GET** `/api/management/customers?limit=50&offset=0`

Returns paginated list of all enrolled customers.

```bash
./scripts/management-list-customers.sh
```

### Get Customer Details

**GET** `/api/management/customers/:customer_id`

Returns full customer details including face metadata.

```bash
./scripts/management-get-customer.sh <customer_id>
```

### Get Statistics

**GET** `/api/management/stats`

Returns database statistics including orphaned faces count.

```bash
./scripts/management-stats.sh
```

**Response:**
```json
{
  "detected_faces": 10,
  "enrolled_customers": 5,
  "orphaned_faces": 5
}
```

### Delete Orphaned Faces

**DELETE** `/api/management/faces/orphaned`

Deletes all orphaned faces (faces not enrolled with any customer) to free up storage space. This includes removing both database records and S3 images.

```bash
./scripts/management-delete-orphaned.sh
```

**Response:**
```json
{
  "message": "Successfully deleted 5 orphaned face(s)",
  "deleted_count": 5,
  "failed_count": 0,
  "deleted_faces": [
    {
      "face_id": "...",
      "identifier": "...",
      "s3_keys": {
        "original": "originals/...",
        "face": "faces/..."
      }
    }
  ]
}
```

### Delete Specific Face

**DELETE** `/api/management/faces/:face_id`

Deletes a specific face and its S3 images (only if not enrolled with any customer).

```bash
./scripts/management-delete-face.sh <face_id>
```

### Delete Customer

**DELETE** `/api/management/customers/:customer_id`

Deletes an enrolled customer record.

```bash
./scripts/management-delete-customer.sh <customer_id>
```

## Example Workflow

```bash
# 1. Detect faces in an image
./scripts/faces-detect.sh examples/elon_musk_enroll.jpg CUSTOMER_001
# Response: [{"face_id": "abc-123...", ...}]

# 2. Enroll the customer
./scripts/faces-enroll.sh abc-123... CUSTOMER_001 "Elon Musk"
# Response: {"customer_id": "xyz-789...", ...}

# 3. Later, detect face in another image of the same person
./scripts/faces-detect.sh examples/elon_musk_positive.jpg
# Response: [{"face_id": "def-456...", ...}]

# 4. Recognize the customer
./scripts/faces-recognize.sh def-456...
# Response: [{"customer_identifier": "CUSTOMER_001", "confidence_score": 0.6708, ...}]

# 5. Retrieve the original face image
./scripts/faces-get-image.sh abc-123...
# Saves image to: output/retrieved_face.jpg
```

## Positive and Negative Recognition Examples

The `examples/` folder includes curated recognition samples:

| Purpose | Image |
|---------|-------|
| Enroll Elon | `examples/elon_musk_enroll.jpg` |
| Positive Elon probe | `examples/elon_musk_positive.jpg` |
| Enroll Jensen | `examples/jensen_huang_enroll.jpg` |
| Positive Jensen probe | `examples/jensen_huang_positive.jpg` |
| Negative probe, should not match enrolled examples | `examples/xi_jinping_solo.png` |
| Multi-format mixed scene | `examples/elon_musk_jensen_huang_mixed_large.webp` |
| AVIF mixed scene | `examples/elon_musk_jensen_huang_mixed_large.avif` |
| Hard profile probe | `examples/elon_musk_profile.jpg` |

Run the smoke test against a running API:

```bash
./scripts/verify-recognition-examples.sh

# Optional: custom API URL and threshold
./scripts/verify-recognition-examples.sh http://localhost:3000 0.5
```

Expected result:
- The positive images return the enrolled Elon and Jensen customers.
- The negative image returns `[]`.
- If the negative image returns a customer, raise `FACE_RECOGNITION_CONFIDENCE_THRESHOLD` and inspect the aligned face image.
- Profile and small mixed-scene examples are intentionally hard cases. Tests use them to make sure weak crops do not become false positives at the default threshold.
- `src/__tests__/examples.test.ts` runs detection against every image file in `examples/` and fails if a new image is added without updating the fixture matrix.

### Performance Benchmark

Run the benchmark against a running API:

```bash
# Terminal 1
make run

# Terminal 2
make benchmark-performance
```

The benchmark measures real API calls for single-face JPEG detection, multi-face JPEG detection, WEBP/AVIF detection, recognition-only lookup, positive detect+recognize flow, and negative detect+recognize flow. It writes JSON and Markdown reports to `benchmarks/results/`, which is ignored by git.

Latest local benchmark sample:

| Field | Value |
|-------|-------|
| Platform | Linux x64 |
| Node.js | v24.16.0 |
| Iterations | 3 |
| ONNX execution provider | CPU |
| GPU acceleration measured | - |
| Database | Local Docker PostgreSQL + pgvector |
| Object storage | Local Docker RustFS |

| Operation | Count | P50 ms | P95 ms | Avg ms |
|-----------|-------|--------|--------|--------|
| `detect_single_face_jpeg` | 3 | 2919.14 | 3297.33 | 2622.87 |
| `detect_multi_face_jpeg` | 3 | 2451.76 | 2817.55 | 2515.09 |
| `detect_webp` | 3 | 8725.34 | 10114.76 | 9164.42 |
| `detect_avif` | 3 | 1450.02 | 1491.38 | 1421.55 |
| `recognize` | 3 | 269.03 | 276.40 | 243.17 |
| `full_positive_flow` | 3 | 2722.87 | 2920.38 | 2720.47 |
| `full_negative_flow` | 3 | 2871.55 | 3492.04 | 3068.68 |

Optional arguments:

```bash
npm run benchmark -- http://localhost:3000 5 benchmarks/results
```

Arguments are `api_url`, `iterations`, and `output_dir`.

## Database Schema

```mermaid
erDiagram
    detected_faces ||--o| enrolled_customers : enrolls

    detected_faces {
        uuid id PK
        text original_image_path
        text face_image_path
        text identifier
        jsonb bounding_box
        float confidence
        timestamptz created_at
    }

    enrolled_customers {
        uuid id PK
        uuid face_id FK
        text customer_identifier
        text customer_name
        jsonb customer_metadata
        vector embedding
        timestamptz created_at
    }
```

### `detected_faces` Table
Stores detected face metadata from the detect endpoint.

| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key (face_id) |
| original_image_path | text | Path to original uploaded image |
| face_image_path | text | Path to cropped face image |
| identifier | text | Optional client identifier |
| bounding_box | jsonb | Face position {x, y, width, height} |
| confidence | float | Detection confidence (0-1) |
| created_at | timestamptz | Timestamp |

### `enrolled_customers` Table
Stores enrolled customer information with face embeddings.

| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key (customer_id) |
| face_id | uuid | Foreign key to detected_faces |
| customer_identifier | text | Unique customer identifier |
| customer_name | text | Customer name (optional) |
| customer_metadata | jsonb | Additional metadata (optional) |
| embedding | vector(512) | InsightFace face embedding |
| created_at | timestamptz | Timestamp |

Vector search is optimized with ivfflat index on the embedding column.

## Image Normalization and Scaling

All uploaded images are normalized through Sharp into auto-rotated JPEG buffers, then scaled down (max 1920px on longest side) to improve processing performance while maintaining aspect ratio. Supports PNG, JPG, WEBP, and AVIF formats up to 10MB.

## Technology Stack

- **Face Detection**: RetinaFace ResNet50 with 5 facial landmarks
- **Face Recognition**: InsightFace Buffalo-L (512-dimensional embeddings)
- **Vector Search**: PostgreSQL pgvector with ivfflat indexing
- **Object Storage**: RustFS S3-compatible storage for images
- **Image Processing**: Jimp for manipulation, multer for uploads
- **API Framework**: Express.js with TypeScript
- **ML Runtime**: ONNX Runtime Node

## Configuration

Environment variables in `.env`:

```bash
# Database
DATABASE_URL=postgres://postgres:postgres@localhost:5432/face_db

# API
PORT=3000

# Face Detection (0.0 - 1.0, default: 0.8)
FACE_DETECTION_CONFIDENCE_THRESHOLD=0.8

# Face Recognition (0.0 - 1.0, default: 0.5)
FACE_RECOGNITION_CONFIDENCE_THRESHOLD=0.5

# RustFS S3-Compatible Storage
S3_ENDPOINT=http://localhost:9000
S3_BUCKET=facevector-engine
S3_ACCESS_KEY=rustfsadmin
S3_SECRET_KEY=rustfsadmin
S3_REGION=us-east-1
S3_FORCE_PATH_STYLE=true
```

## Storage Structure

### RustFS Bucket (facevector-engine)
```
facevector-engine/
├── originals/          # Normalized uploaded images
│   └── {uuid}.jpg
└── faces/              # Cropped face images
    └── {face_id}.jpg
```

### Temporary Files (Docker Container)
```
/tmp/facevector/
└── cropped_faces/      # Temporary cropped faces during processing
    └── face_*.jpg      # Auto-cleaned on container restart
```

**RustFS Console**: Access at http://localhost:9001 with credentials from `.env`

## Makefile Commands

### Development
```bash
make install        # Install dependencies (production only)
make install-dev    # Install all dependencies including dev
make models         # Download ONNX models
make chmod-scripts  # Make scripts executable
make run            # Run API locally (starts db + RustFS)
make lint           # Check code style
make lint-fix       # Auto-fix code style
npm run build       # Compile TypeScript
```

### Testing
```bash
make test              # Run full test suite (starts db + RustFS)
make test-integration  # Run integration tests (full workflow)
make test-management   # Run management API tests
make test-examples     # Run every examples/ image through detection
make test-storage      # Verify RustFS bucket/object operations
make benchmark-performance  # Benchmark a running API and write reports
```

### Docker
```bash
make up             # Build and start full stack (API + db + RustFS)
make down           # Stop containers
make logs           # Tail container logs
make clean          # Remove containers and volumes
make reset          # Clean and rebuild from scratch
make up-db          # Start PostgreSQL only
make up-rustfs      # Start RustFS only
```

## Troubleshooting

### Database Collation Errors
```bash
make clean && make up
```

### Face Not Detected
1. Lower `FACE_DETECTION_CONFIDENCE_THRESHOLD` in `.env` (e.g., 0.6)
2. Ensure clear, front-facing faces in images
3. Check image format (PNG, JPG, WEBP, AVIF only)

### ONNX Models Not Found
```bash
make models  # Downloads to models/ directory
```

## License

[MIT License](LICENSE) - free to use for any purpose.
