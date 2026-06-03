# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FaceVector Engine - A production-ready face recognition and vector similarity search engine. A Node.js/TypeScript system using InsightFace Buffalo-L embeddings for face recognition, RetinaFace for face detection with landmarks, and PostgreSQL with pgvector extension for efficient vector similarity search.

**📚 Technical Documentation:** See [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md) for comprehensive technical documentation including image processing pipelines, coordinate transformations, model inference workflows, and performance optimizations.

**API Design**: Two-step workflow for face recognition:
1. **Detect**: Upload image → Detect faces → Get face_id
2. **Enroll or Recognize**: Use face_id to enroll customer or recognize against enrolled database

## Development Commands

### Setup
```bash
make install          # Install Node dependencies (production only)
make install-dev      # Install all dependencies including dev tools
make models           # Download ONNX models (face_recognition.onnx, retinaface_resnet50.onnx) to models/
make chmod-scripts    # Make scripts in scripts/ executable
```

**Model Downloads** (`make models`):
- InsightFace Buffalo-L (`w600k_r50.onnx`) from HuggingFace → saved as `models/face_recognition.onnx`
- RetinaFace ResNet50 from Google Storage → saved as `models/retinaface_resnet50.onnx`
- Models only downloaded if they don't already exist

### Running Locally
```bash
make db               # Start PostgreSQL only (detached, port 5432)
make run              # Run API with tsx watch (requires db running)
                      # Sets DATABASE_URL=postgres://postgres:postgres@localhost:5432/face_db
                      # API runs on http://localhost:3000/api
```

### Docker
```bash
make up               # Build and start full stack (API + Postgres)
make down             # Stop containers
make logs             # Tail container logs
make clean            # Remove containers and volumes
make reset            # Clean and rebuild from scratch
```

### Build & Lint
```bash
npm run build         # Compile TypeScript to dist/
npm run dev           # Watch mode with tsx
npm start             # Run compiled dist/server.js
make lint             # Check TypeScript code style
make lint-fix         # Auto-fix TypeScript code style issues
```

## Architecture

### Core Flow
1. **Server startup** (`src/server.ts`): Initialize DB connection → Load ONNX models (InsightFace + RetinaFace) → Start Express server
2. **Face detection** (`src/retinaface.ts`, `src/embedding.ts`): RetinaFace detects faces with configurable thresholds
3. **Embedding generation** (`src/embedding.ts`): InsightFace generates 512-dim vectors from landmark-aligned 112x112 BGR images
4. **Storage** (`src/db.ts`): PostgreSQL with pgvector extension for vector similarity search
   - `detected_faces` table: Face metadata and file paths from detection
   - `enrolled_customers` table: Customer info with face embeddings for recognition
   - `face_embeddings` table: Legacy table (kept for backward compatibility)

### Key Components

**ONNX Model Pipeline**:
- `initModels()` (`src/embedding.ts`): Loads InsightFace (embedding) and RetinaFace (detection) ONNX models at startup
- `detectAllFacesWithRetinaFace(base64)` (`src/embedding.ts`): RetinaFace inference returning all detected faces with bounding boxes
  - Input: 640x640 (mobile) or 840x840 (resnet50) resized image
  - Multi-scale detection with strides [8, 16, 32]
  - Default confidence threshold: 0.8 (VIS_THRESHOLD), configurable via env var FACE_DETECTION_CONFIDENCE_THRESHOLD
  - NMS (Non-Maximum Suppression) threshold: 0.4 to filter overlapping boxes
  - Returns array of faces sorted by area (largest first) with landmarks (eyes, nose, mouth)
- `preprocessImage(base64)` (`src/embedding.ts`): Resize to 112x112, BGR channel-first layout, normalize to [-1, 1] per channel
- `computeEmbedding(preprocessed)` (`src/embedding.ts`): InsightFace inference → 512-dim Float32Array
- RetinaFace model (`src/retinaface.ts`): Detects faces with 5 facial landmarks per face

**Database Layer** (`src/db.ts`):
- `connectDB()`: Creates database if missing, enables pgcrypto + vector extensions
- Table schemas:
  - `detected_faces(id uuid, original_image_path text, face_image_path text, identifier text, bounding_box jsonb, confidence float, created_at timestamptz)`
  - `enrolled_customers(id uuid, face_id uuid, customer_identifier text, customer_name text, customer_metadata jsonb, embedding vector(512), created_at timestamptz)`
  - `face_embeddings(id uuid, embedding vector(512), image_path text, created_at timestamptz)` - Legacy table
- ivfflat index on `enrolled_customers.embedding` for fast similarity search
- Handles collation version mismatches (suggests `make clean` if detected)
- Images stored in RustFS S3-compatible object storage, S3 keys stored in `original_image_path` and `face_image_path` columns

**Service Layer**:
- `src/services/faceDetectionService.ts`: Detect faces, crop, and store metadata
  - `detectAndStoreFaces()`: Main detection workflow
  - Uploads normalized original images to RustFS S3: `originals/{uuid}.jpg`
  - Uploads cropped faces to RustFS S3: `faces/{face_id}.jpg`
  - Stores S3 keys (not file paths) in database
- `src/services/s3Service.ts`: RustFS S3-compatible storage service
  - `uploadImage()`: Upload image buffer to S3
  - `downloadImage()`: Download image from S3 as buffer
  - `deleteImage()`: Delete image from S3
  - `ensureBucketExists()`: Create bucket if missing
- `src/services/imageService.ts`: Legacy service for base64 image processing
- `src/services/cropFacesService.ts`: Extract cropped face images from detected bounding boxes
- `src/services/dbService.ts`: Legacy vector insert/search operations

**Utils Layer** (`src/utils/`):
- `imageUtils.ts`: Image manipulation
  - `base64ToJimp()`: Convert base64 to Jimp instance
  - `bufferToJimp()`: Convert Buffer (from multer) to Jimp instance
  - `jimpToBase64()`: Convert Jimp to base64
  - `scaleDownImage()`: Scale down images for performance (max 1920px)
  - `cropImageRegion()`: Crop face regions
- `boundingBoxUtils.ts`: Coordinate conversions (normalized ↔ pixel bounding boxes)
- `nmsUtils.ts`: Non-Maximum Suppression for filtering overlapping face detections
- `visualizationUtils.ts`: Draw bounding boxes and landmarks on images
- `responseHelpers.ts`: Standardized error responses

**Middleware**:
- `src/middleware/upload.ts`: Multer configuration for multipart file uploads
  - Memory storage for file buffers
  - Accepts PNG, JPG, WEBP, AVIF (max 10MB)
  - File type validation

**Controllers** (`src/controllers/`):
- `facesController.ts`: Main API endpoints
  - `detectFaces()`: POST `/faces/detect` - Detect faces, store crops and metadata
  - `getFaceImage()`: GET `/faces/:face_id` - Retrieve stored face image
  - `enrollFace()`: POST `/faces/enroll` - Enroll customer with face embedding
  - `recognizeFace()`: POST `/faces/recognize` - Recognize customer by face similarity
- `managementController.ts`: Management/admin endpoints
  - `listDetectedFaces()`: GET `/management/faces` - Paginated list of detected faces
  - `listEnrolledCustomers()`: GET `/management/customers` - Paginated list of customers
  - `getCustomerDetails()`: GET `/management/customers/:id` - Customer details
  - `getStats()`: GET `/management/stats` - DB stats (includes orphaned_faces count)
  - `deleteOrphanedFaces()`: DELETE `/management/faces/orphaned` - Delete orphaned faces + S3 cleanup
  - `deleteDetectedFace()`: DELETE `/management/faces/:id` - Delete specific face + S3 cleanup
  - `deleteEnrolledCustomer()`: DELETE `/management/customers/:id` - Delete customer
- Legacy controllers (backward compatibility):
  - `faceDetectionController.ts`, `faceVisualizationController.ts`, `cropFacesController.ts`, `embeddingController.ts`, `imageController.ts`

**Routes** (`src/routes.ts`):
- POST `/api/faces/detect` - Upload image → Detect faces → Store → Return face_id array
- GET `/api/faces/:face_id` - Retrieve face image
- POST `/api/faces/enroll` - Enroll customer with embedding
- POST `/api/faces/recognize` - Recognize customer by similarity
- GET `/api/management/faces` - List detected faces (pagination)
- GET `/api/management/customers` - List customers (pagination)
- GET `/api/management/customers/:id` - Get customer details
- GET `/api/management/stats` - Database statistics
- DELETE `/api/management/faces/orphaned` - Delete orphaned faces (not enrolled)
- DELETE `/api/management/faces/:id` - Delete specific face
- DELETE `/api/management/customers/:id` - Delete customer

### Error Handling
- All face-processing endpoints throw `{code: "NO_FACE"}` errors → Returns `{"error": "no_face_detected"}` (400)
- Face detection configured to reject images without faces (e.g., `examples/no_face_box.jpeg`)
- Standardized error responses via `responseHelpers.sendErrorResponse()`

### API Input Format & Storage

**Multipart Upload Design:**
- Primary API accepts multipart form-data (actual file uploads)
- Scripts handle file path → multipart conversion automatically
- Images normalized through Sharp into auto-rotated JPEG buffers, then scaled down (max 1920px) for performance
- Supports PNG, JPG, WEBP, AVIF formats (max 10MB per file)
- API is stateless and production-ready

**ONNX Models:**
- Face recognition: `models/face_recognition.onnx` (InsightFace embeddings)
- RetinaFace: `models/retinaface_resnet50.onnx` (face detection)
- Downloaded via `make models` from HuggingFace/Google Storage

**Storage Architecture:**
- **RustFS S3-Compatible Object Storage** (bucket: `facevector-engine`):
  - `originals/{uuid}.jpg` - Normalized uploaded images
  - `faces/{face_id}.jpg` - Cropped face images
  - S3 keys stored in `detected_faces` table columns
  - Access RustFS Console at http://localhost:9001
- **Temporary Files** (ephemeral, inside Docker):
  - `/tmp/facevector/cropped_faces/` - Temporary face crops during processing
  - Auto-cleaned on container restart, no volume mount needed

## Testing
Use example images in `examples/` folder:
- `elon_musk_enroll.jpg`, `elon_musk_positive.jpg` - Valid Elon images for testing (same person)
- `jensen_huang_enroll.jpg`, `jensen_huang_positive.jpg` - Valid Jensen images for testing (same person)
- `elon_musk_trump_mixed_small.jpg`, `xi_jinping_trump_mixed_small.jpeg` - Multiple faces in one image
- `elon_musk_jensen_huang_mixed_large.webp`, `elon_musk_jensen_huang_mixed_large.avif` - Multi-format normalization fixtures
- `elon_musk_profile.jpg`, `trump_jensen_huang_mixed_profile.jpeg` - Hard profile/angle examples
- `xi_jinping_solo.png` - Additional negative test image
- `no_face_box.jpeg` - No face (for testing detection rejection)

Example scripts in `scripts/` (run `make chmod-scripts` first):
```bash
# Main workflow
./scripts/faces-detect.sh examples/elon_musk_enroll.jpg [identifier]
./scripts/faces-get-image.sh <face_id>
./scripts/faces-enroll.sh <face_id> <customer_id> [name]
./scripts/faces-recognize.sh <face_id>

# Management
./scripts/management-list-faces.sh
./scripts/management-list-customers.sh
./scripts/management-get-customer.sh <customer_id>
./scripts/management-stats.sh
./scripts/management-delete-orphaned.sh              # Delete orphaned faces
./scripts/management-delete-face.sh <face_id>
./scripts/management-delete-customer.sh <customer_id>
```

All scripts accept optional API URL as last parameter (default: http://localhost:3000).

**Workflow Example:**
```bash
# 1. Detect face and get face_id
./scripts/faces-detect.sh examples/elon_musk_enroll.jpg CUST001
# Response: [{"face_id": "abc-123...", ...}]

# 2. Enroll the customer
./scripts/faces-enroll.sh abc-123... CUST001 "Elon Musk"
# Response: {"customer_id": "xyz-789...", ...}

# 3. Later, detect face in another image of the same person
./scripts/faces-detect.sh examples/elon_musk_positive.jpg
# Response: [{"face_id": "def-456...", ...}]

# 4. Recognize who it is
./scripts/faces-recognize.sh def-456...
# Response: [{"customer_identifier": "CUST001", "confidence_score": 0.6708, ...}]
```

## Configuration

**Environment Variables** (`.env`):
- `DATABASE_URL`: PostgreSQL connection string (default: `postgres://postgres:postgres@localhost:5432/face_db`)
- `PORT`: API server port (default: 3000)
- `FACE_DETECTION_CONFIDENCE_THRESHOLD`: Face detection confidence threshold 0.0-1.0 (default: 0.8)
- **RustFS S3 Configuration:**
  - `S3_ENDPOINT`: S3 endpoint URL (default: http://localhost:9000 for local, http://rustfs:9000 inside Docker)
  - `S3_BUCKET`: S3 bucket name (default: facevector-engine)
  - `S3_ACCESS_KEY`: S3 access key (default: rustfsadmin)
  - `S3_SECRET_KEY`: S3 secret key (default: rustfsadmin)
  - `S3_REGION`: S3 region (default: us-east-1)
  - `S3_FORCE_PATH_STYLE`: Use path-style URLs for RustFS (default: true)

**Configuration Constants** (`src/config/constants.ts`):
- `RETINAFACE.CONFIDENCE_THRESHOLD`: 0.02 (initial detection threshold, filtered by VIS_THRESHOLD later)
- `RETINAFACE.VIS_THRESHOLD`: 0.8 (final visibility threshold, overridden by env var FACE_DETECTION_CONFIDENCE_THRESHOLD)
- `RETINAFACE.NMS_THRESHOLD`: 0.4 (Non-Maximum Suppression for overlapping boxes)
- `RETINAFACE.TOP_K`: 5000 (maximum detections before NMS)
- `RETINAFACE.KEEP_TOP_K`: 750 (maximum detections after NMS)
- `PATHS.TEMP_DIR`: "/tmp/facevector" (temporary processing inside Docker)
- `PATHS.CROPPED_FACES_DIR`: "/tmp/facevector/cropped_faces" (temporary cropped faces)
- `PATHS.MODELS_DIR`: "models" (ONNX models directory)
- `S3_CONFIG.BUCKET`: process.env.S3_BUCKET || "facevector-engine"
- `S3_CONFIG.ORIGINALS_PREFIX`: "originals/" (S3 prefix for original images)
- `S3_CONFIG.FACES_PREFIX`: "faces/" (S3 prefix for cropped face images)
- Image sizes: 640x640 (mobile0.25) or 840x840 (resnet50)

## Database Notes
- DATABASE_URL env var required (format: `postgres://user:pass@host:port/dbname`)
- Database auto-created on first connection if missing
- pgvector extension required for vector operations
- ivfflat index created on `enrolled_customers.embedding` for fast similarity search
- If collation errors occur during `make up`, run `make clean` then `make up` to reset volumes

## Dependencies

**Core Runtime:**
- `express`: Web framework
- `pg`: PostgreSQL client
- `pgvector`: PostgreSQL vector extension client
- `onnxruntime-node`: ML model inference
- `jimp`: Image manipulation
- `multer`: Multipart file upload handling
- `@types/multer`: TypeScript types for multer

**Development:**
- `typescript`: TypeScript compiler
- `tsx`: TypeScript execution for development
- `eslint`: Code linting
- `@types/*`: TypeScript type definitions
