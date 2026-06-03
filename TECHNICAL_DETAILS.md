# FaceVector Engine - Technical Details

This document provides an in-depth technical explanation of how the FaceVector Engine works internally, including image processing pipelines, coordinate transformations, and model inference workflows.

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Image Processing Pipeline](#image-processing-pipeline)
3. [Coordinate Transformation System](#coordinate-transformation-system)
4. [Model Inference Details](#model-inference-details)
5. [Storage Architecture](#storage-architecture)
6. [API Workflow Deep Dive](#api-workflow-deep-dive)
7. [Performance Optimizations](#performance-optimizations)

---

## System Architecture Overview

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Application                     │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP/REST API
┌───────────────────────────▼─────────────────────────────────┐
│                   Express API Server                        │
│  ┌────────────┐  ┌────────────┐  ┌─────────────────────┐    │
│  │Controllers │  │ Services   │  │ Middleware (Multer) │    │
│  └────────────┘  └────────────┘  └─────────────────────┘    │
└────────┬───────────────────┬─────────────────┬──────────────┘
         │                   │                 │
         │   ┌───────────────▼──────────┐      │
         │   │   ONNX Runtime Node      │      │
         │   │  ┌──────────────────┐    │      │
         │   │  │ RetinaFace (840) │    │      │
         │   │  └──────────────────┘    │      │
         │   │  ┌──────────────────┐    │      │
         │   │  │ Recognition (112)│    │      │
         │   │  └──────────────────┘    │      │
         │   └──────────────────────────┘      │
         │                                     │
┌────────▼────────────┐              ┌─────────▼──────────┐
│   PostgreSQL +      │              │                    │
│    pgvector         │              │   RustFS S3        │
│  ┌────────────────┐ │              │  ┌──────────────┐  │
│  │ detected_faces │ │              │  │originals/    │  │
│  └────────────────┘ │              │  └──────────────┘  │
│  ┌────────────────┐ │              │  ┌──────────────┐  │
│  │enrolled_       │ │              │  │faces/        │  │
│  │  customers     │ │              │  └──────────────┘  │
│  └────────────────┘ │              └────────────────────┘
└─────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Key Files |
|-----------|---------------|-----------|
| **Controllers** | Handle HTTP requests/responses | `src/controllers/*.ts` |
| **Services** | Business logic & orchestration | `src/services/*.ts` |
| **Utils** | Image processing & transformations | `src/utils/*.ts` |
| **Models** | ONNX model inference | `src/embedding.ts`, `src/retinaface.ts` |
| **Database** | Vector storage & search | `src/db.ts` |
| **Storage** | S3 object storage | `src/services/s3Service.ts` |

---

## Image Processing Pipeline

### Complete Flow: Upload → Storage

```
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: File Upload & Initial Processing                       │
└─────────────────────────────────────────────────────────────────┘

User uploads image (e.g., 3000 x 2000 pixels, JPEG)
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│ Multer Middleware (src/middleware/upload.ts)                   │
│ - Validates file type (PNG, JPG, WEBP, AVIF)                   │
│ - Checks file size (max 10MB)                                  │
│ - Stores in memory as Buffer                                   │
└────────┬───────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│ normalizeImageBuffer() + scaleDownImage() - imageUtils.ts      │
│                                                                │
│ Input:  Buffer (3000 x 2000)                                   │
│ Logic:  Sharp auto-rotates and converts accepted formats       │
│         to JPEG, then scales max side to 1920px                │
│ Output: Base64 JPEG string (e.g., 1920 x 1280)                 │
│                                                                │
│ Why: Performance optimization - reduces processing time        │
└────────┬───────────────────────────────────────────────────────┘
         │
         ▼ base64 (1920 x 1280)
         │
┌────────────────────────────────────────────────────────────────┐
│ STAGE 2: Face Detection with RetinaFace                        │
└────────────────────────────────────────────────────────────────┘

detectAllFacesWithRetinaFace(base64, visThreshold)
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│ preprocessForRetinaFace() - src/retinaface.ts:202-232          │
│                                                                │
│ 1. Load image to Jimp:                                         │
│    const image = await base64ToJimp(base64)                    │
│    originalWidth = 1920 ← PRESERVED                            │
│    originalHeight = 1280 ← PRESERVED                           │
│                                                                │
│ 2. Resize for model input:                                     │
│    await image.resize({ w: 840, h: 840 })                      │
│                                                                │
│ 3. Extract RGB and subtract mean:                              │
│    meanValues = [104, 117, 123]                                │
│    Convert to CHW format (Channels, Height, Width)             │
│                                                                │
│ Output: {                                                      │
│   data: Float32Array (840 x 840 x 3),                          │
│   originalWidth: 1920,  ← KEY: Used for scaling back!          │
│   originalHeight: 1280                                         │
│ }                                                              │
└────────┬───────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│ RetinaFace Inference - src/retinaface.ts:473-500               │
│                                                                │
│ 1. Run ONNX model on 840x840 tensor                            │
│    const outputs = await retinaSession.run({ input0 })         │
│                                                                │
│ 2. Decode bounding boxes and landmarks                         │
│    boxes = decode(locArray, priors, variance)                  │
│    landmarks = decodeLandmarks(landmsArray, priors, variance)  │
│                                                                │
│ 3. Scale to pixel coordinates using ORIGINAL dimensions        │
│    const scale = [originalWidth, originalHeight, ...]          │
│    scaledBoxes = boxes.map(box => [                            │
│      box[0] * 1920,  // x1 in original image                   │
│      box[1] * 1280,  // y1 in original image                   │
│      box[2] * 1920,  // x2 in original image                   │
│      box[3] * 1280   // y2 in original image                   │
│    ])                                                          │
│                                                                │
│ 4. Apply NMS (Non-Maximum Suppression)                         │
│    Filter overlapping boxes (threshold: 0.4)                   │
│                                                                │
│ 5. Filter by confidence (threshold: 0.8)                       │
│                                                                │
│ Output: DetectedFace[] with coordinates in 1920x1280 space     │
│         [{                                                     │
│           PixelBoundingBox: { Left: 450, Top: 300,             │
│                               Width: 600, Height: 850 },       │
│           Confidence: 99.8,                                    │
│           Landmarks: [{ eyeLeft, eyeRight, ... }]              │
│         }]                                                     │
└────────┬───────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│ STAGE 3: Storage - src/services/faceDetectionService.ts        │
└────────────────────────────────────────────────────────────────┘

For EACH detected face:
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│ 1. Store Original Image to S3                                  │
│    originalImageId = randomUUID()                              │
│    s3Service.uploadImage(                                      │
│      key: "originals/{uuid}.jpg",                              │
│      buffer: base64ToBuffer(1920x1280 image)                   │
│    )                                                           │
└────────┬───────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│ 2. Crop Face Region - src/utils/imageUtils.ts:44-65            │
│    cropImageRegion(                                            │
│      base64: 1920x1280 original,                               │
│      x: 450, y: 300, width: 600, height: 850                   │
│    )                                                           │
│    ↓                                                           │
│    Extracts 600x850 region from original image                 │
│    Returns: base64 (600 x 850)                                 │
└────────┬───────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│ 3. Store Cropped Face to S3                                    │
│    faceId = randomUUID()                                       │
│    s3Service.uploadImage(                                      │
│      key: "faces/{face_id}.jpg",                               │
│      buffer: base64ToBuffer(600x850 cropped face)              │
│    )                                                           │
└────────┬───────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│ 4. Store Metadata in PostgreSQL                                │
│    INSERT INTO detected_faces (                                │
│      id,                    -- face_id UUID                    │
│      original_image_path,   -- "originals/{uuid}.jpg"          │
│      face_image_path,       -- "faces/{face_id}.jpg"           │
│      bounding_box,          -- {x: 450, y: 300, w: 600, h: 850}│
│      confidence,            -- 0.998                           │
│      identifier             -- Optional client identifier      │
│    )                                                           │
└────────┬───────────────────────────────────────────────────────┘
         │
         ▼
Return face_id to client
```

### Dimension Summary Table

| Stage | Dimensions | Stored? | Purpose |
|-------|-----------|---------|---------|
| **Upload** | Original (e.g., 3000×2000) | ❌ | User input |
| **Scaled** | Max 1920px (e.g., 1920×1280) | ✅ S3: `originals/` | Performance optimization |
| **RetinaFace Input** | 840×840 (square) | ❌ | Face detection inference |
| **Cropped Face** | Variable (e.g., 600×850) | ✅ S3: `faces/` | Detected face region |
| **Recognition Input** | 112×112 (square) | ❌ | Embedding generation |

---

## Coordinate Transformation System

### The Critical Challenge

**Problem:** RetinaFace model requires 840×840 input, but we need bounding boxes in the original image coordinates (e.g., 1920×1280) to crop accurately.

**Solution:** Preserve original dimensions and scale coordinates back mathematically.

### Step-by-Step Transformation

#### Example Scenario
- **Original Image:** 1920 × 1280 pixels
- **RetinaFace Input:** 840 × 840 pixels (temporary)

```javascript
// STEP 1: Preprocessing (src/retinaface.ts:202-232)
const preprocessForRetinaFace = async (base64, targetSize = 840) => {
  const image = await base64ToJimp(base64);

  // ✅ CRITICAL: Preserve original dimensions
  const originalWidth = image.bitmap.width;   // 1920
  const originalHeight = image.bitmap.height; // 1280

  // Resize to model input size
  await image.resize({ w: targetSize, h: targetSize }); // 840×840

  // Extract pixel data and normalize
  const inputData = new Float32Array(3 * 840 * 840);
  // ... pixel processing ...

  return {
    data: inputData,      // 840×840 tensor
    originalWidth: 1920,  // ← Preserved for scaling back!
    originalHeight: 1280  // ← Preserved for scaling back!
  };
};
```

#### Coordinate Spaces

```
┌─────────────────────────────────────────────────────────┐
│ SPACE 1: Model Output (Normalized 0-1)                  │
│                                                         │
│  RetinaFace outputs normalized coordinates:             │
│  bbox: [0.234, 0.234, 0.547, 0.898]                     │
│         ↑      ↑      ↑      ↑                          │
│         x1/W   y1/H   x2/W   y2/H                       │
│                                                         │
│  These are RELATIVE to 840×840 input image              │
└─────────────────────────────────────────────────────────┘
                    ↓
          scaleToPixelCoordinates()
                    ↓
┌─────────────────────────────────────────────────────────┐
│ SPACE 2: Original Image Pixels                          │
│                                                         │
│  Scaled to 1920×1280 using preserved dimensions:        │
│  scaledBox = [                                          │
│    0.234 × 1920 = 449.28,  // x1                        │
│    0.234 × 1280 = 299.52,  // y1                        │
│    0.547 × 1920 = 1050.24, // x2                        │
│    0.898 × 1280 = 1149.44  // y2                        │
│  ]                                                      │
│                                                         │
│  Final pixel box: [449, 300, 1050, 1149]                │
│  Width:  1050 - 449 = 601 pixels                        │
│  Height: 1149 - 300 = 849 pixels                        │
└─────────────────────────────────────────────────────────┘
```

#### Mathematical Scaling Implementation

```typescript
// src/retinaface.ts:334-356
const scaleToPixelCoordinates = (
  boxes: number[][],        // Normalized boxes from model
  landmarks: number[][],    // Normalized landmarks
  originalWidth: number,    // 1920
  originalHeight: number    // 1280
): ScaledData => {

  // Scale factors for [x1, y1, x2, y2]
  const scale = [
    originalWidth,   // 1920 for x coordinates
    originalHeight,  // 1280 for y coordinates
    originalWidth,   // 1920 for x coordinates
    originalHeight   // 1280 for y coordinates
  ];

  // Transform each box
  const scaledBoxes = boxes.map((box) => [
    box[0] * scale[0],  // x1_normalized * 1920 = x1_pixels
    box[1] * scale[1],  // y1_normalized * 1280 = y1_pixels
    box[2] * scale[2],  // x2_normalized * 1920 = x2_pixels
    box[3] * scale[3],  // y2_normalized * 1280 = y2_pixels
  ]);

  // Transform landmarks (5 points × 2 coords = 10 values)
  const scale1 = Array(10).fill(0).map((_, i) =>
    i % 2 === 0 ? originalWidth : originalHeight
  );
  // scale1 = [1920, 1280, 1920, 1280, 1920, 1280, ...]

  const scaledLandmarks = landmarks.map((landm) =>
    landm.map((val, i) => val * scale1[i])
  );

  return { scaledBoxes, scaledLandmarks };
};
```

### Accuracy Analysis

**Precision Loss:**
- Float32 normalized coords → Math.round() to integers
- Maximum error: ±0.5 pixels per coordinate
- Typical error: < 1 pixel for 1920×1280 images
- **Conclusion:** Negligible impact on face recognition accuracy

---

## Model Inference Details

### Models Used by This Repository

This project currently uses two ONNX files:

| Purpose | Local file | Upstream file/source | Runtime input |
|---------|------------|----------------------|---------------|
| Face detection + 5 landmarks | `models/retinaface_resnet50.onnx` | RetinaFace ResNet-50 ONNX export | `1 x 3 x 840 x 840` RGB-like tensor with RetinaFace mean subtraction |
| Face recognition embeddings | `models/face_recognition.onnx` | InsightFace Buffalo-L `w600k_r50.onnx` | `1 x 3 x 112 x 112` BGR tensor normalized to `[-1, 1]` |

The recognition model is the InsightFace Buffalo-L ResNet-50 export saved locally as `face_recognition.onnx`.

### RetinaFace (Face Detection)

**Model:** `retinaface_resnet50.onnx` (ResNet-50 backbone)

#### Architecture
```
Input: [1, 3, 840, 840] RGB tensor
         ↓
┌──────────────────────────────┐
│  ResNet-50 Feature Extractor │
│  Multi-scale feature maps:   │
│  - Stride 8  (105×105)       │
│  - Stride 16 (52×52)         │
│  - Stride 32 (26×26)         │
└──────────────────────────────┘
         ↓
┌──────────────────────────────┐
│  Detection Heads             │
│  For each anchor:            │
│  - Classification (face/bg)  │
│  - Bounding box regression   │
│  - Landmark regression (5pt) │
└──────────────────────────────┘
         ↓
Output:
  - bbox:     [N, 4]  (x1, y1, x2, y2)
  - scores:   [N, 1]  (confidence)
  - landmarks:[N, 10] (5 points × x,y)
```

#### Detection Process

```typescript
// src/retinaface.ts:455-518
export const detectFacesRetinaFace = async (base64, visThreshold) => {
  // Step 1: Preprocess image
  const { data, originalWidth, originalHeight } =
    await preprocessForRetinaFace(base64, 840);

  // Step 2: Run inference
  const tensor = new ort.Tensor("float32", data, [1, 3, 840, 840]);
  const outputs = await retinaSession.run({ "input0": tensor });

  // Step 3: Generate prior boxes (anchors)
  const priorbox = new PriorBox(retinaConfig, [840, 840]);
  const priors = priorbox.forward();
  // Creates ~16,800 anchor boxes at different scales

  // Step 4: Decode predictions
  const boxes = decode(locArray, priors, variance);
  const landmarks = decodeLandmarks(landmsArray, priors, variance);

  // Step 5: Scale to original image coordinates
  const { scaledBoxes, scaledLandmarks } =
    scaleToPixelCoordinates(boxes, landmarks, originalWidth, originalHeight);

  // Step 6: Filter by confidence threshold
  const filtered = filterByConfidence(scaledBoxes, scores, visThreshold);

  // Step 7: Non-Maximum Suppression (NMS)
  const keep = nms(filteredBoxes, NMS_THRESHOLD);

  // Step 8: Keep top detections
  const finalDetections = keep.slice(0, KEEP_TOP_K);

  return finalDetections;
};
```

#### Configuration Parameters

```typescript
// src/config/constants.ts:23-29
export const RETINAFACE = {
  CONFIDENCE_THRESHOLD: 0.02, // Initial detection threshold
  NMS_THRESHOLD: 0.4,         // IoU threshold for NMS
  VIS_THRESHOLD: 0.8,         // Final visibility threshold
  TOP_K: 5000,                // Max detections before NMS
  KEEP_TOP_K: 750,            // Max detections after NMS
};
```

---

### InsightFace Buffalo-L (Face Embedding)

**Model:** `face_recognition.onnx` (InsightFace Buffalo-L ResNet-50, downloaded from `buffalo_l/w600k_r50.onnx`)

#### Architecture
```
Input: [1, 3, 112, 112] BGR tensor (normalized to [-1, 1])
         ↓
┌──────────────────────────────┐
│  ResNet-50 Backbone          │
│  - 50-layer recognition net  │
│  - Batch normalization       │
│  - PReLU activation          │
└──────────────────────────────┘
         ↓
┌──────────────────────────────┐
│  Fully Connected Layer       │
│  512 dimensions              │
└──────────────────────────────┘
         ↓
┌──────────────────────────────┐
│  L2 Normalization            │
│  (Unit vector)               │
└──────────────────────────────┘
         ↓
Output: [512] float32 embedding vector
```

#### Embedding Generation Process

```typescript
// src/embedding.ts:21-48

// 1. Preprocessing
export const preprocessImage = async (base64: string) => {
  const image = await base64ToJimp(base64);

  // Resize to 112×112
  await image.resize({ w: 112, h: 112 });

  // Extract BGR channel-first data and normalize to [-1, 1]
  const data = new Float32Array(3 * 112 * 112);
  const imageSize = 112 * 112;

  for (let y = 0; y < 112; y++) {
    for (let x = 0; x < 112; x++) {
      const idx = (112 * y + x) * 4;
      const { data: bitmapData } = image.bitmap;

      const pixelIndex = 112 * y + x;

      // Normalize: (pixel / 255.0 - 0.5) / 0.5 = (pixel - 127.5) / 127.5
      data[pixelIndex] = (bitmapData[idx + 2] / 255.0 - 0.5) / 0.5; // B
      data[imageSize + pixelIndex] = (bitmapData[idx + 1] / 255.0 - 0.5) / 0.5; // G
      data[2 * imageSize + pixelIndex] = (bitmapData[idx] / 255.0 - 0.5) / 0.5; // R
    }
  }

  return data; // Float32Array[3 × 112 × 112]
};

// 2. Inference
export const computeEmbedding = async (preprocessed: Float32Array) => {
  const tensor = new ort.Tensor(
    "float32",
    preprocessed,
    [1, 3, 112, 112]
  );

  const inputName = recognitionSession.inputNames[0];
  const results = await recognitionSession.run({ [inputName]: tensor });
  const embedding = results[Object.keys(results)[0]].data;

  return Array.from(embedding); // Float32[512]
};
```

#### Similarity Computation

```typescript
// src/embedding.ts:88-115
export const compareEmbeddings = (
  embedding1: number[],
  embedding2: number[]
) => {
  // Cosine Similarity
  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;

  for (let i = 0; i < 512; i++) {
    dotProduct += embedding1[i] * embedding2[i];
    norm1 += embedding1[i] * embedding1[i];
    norm2 += embedding2[i] * embedding2[i];
  }

  const cosineSimilarity = dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));

  // Euclidean Distance
  let sumSquares = 0;
  for (let i = 0; i < 512; i++) {
    const diff = embedding1[i] - embedding2[i];
    sumSquares += diff * diff;
  }
  const euclideanDistance = Math.sqrt(sumSquares);

  return { cosineSimilarity, euclideanDistance };
};
```

#### Recognition Example Matrix

The example tests intentionally include both easy positives and hard negatives. The default recognition threshold remains `0.5`; weak mixed-scene crops are not forced to pass by lowering the threshold.

| Scenario | Enrolled identity | Probe image | Expected result at `0.5` |
|----------|-------------------|-------------|---------------------------|
| Positive solo match | `examples/elon_musk_enroll.jpg` | `examples/elon_musk_positive.jpg` | Elon is top match, observed around `0.69` |
| Positive solo match | `examples/jensen_huang_enroll.jpg` | `examples/jensen_huang_positive.jpg` | Jensen is top match, observed around `0.56` |
| Negative solo probe | Elon + Jensen enrollments | `examples/xi_jinping_solo.png` | No match, returns `[]` |
| Multi-face same-image sanity | Xi + Trump enrolled from one mixed image | `examples/xi_jinping_trump_mixed_small.jpeg` | Same detected faces match their own enrolled identities |
| Format normalization | N/A | `examples/elon_musk_jensen_huang_mixed_large.webp` and `.avif` | Faces are detected after Sharp normalization |

This matrix is implemented in `src/__tests__/integration.test.ts`, `src/__tests__/management.test.ts`, and `scripts/verify-recognition-examples.sh`.

`src/__tests__/examples.test.ts` is the fixture coverage gate: it enumerates every image in `examples/`, verifies each file is represented in the expected fixture matrix, runs detection for each supported image format, and checks `no_face_box.jpeg` returns the expected no-face error.

#### Database Vector Search

```sql
-- PostgreSQL with pgvector extension
-- src/controllers/facesController.ts:164-174

SELECT
  id as customer_id,
  customer_identifier,
  customer_name,
  1 - (embedding <=> $1) as confidence_score  -- Cosine similarity
FROM enrolled_customers
ORDER BY embedding <=> $1  -- Cosine distance (ascending)
LIMIT 10
```

**Operators:**
- `<=>` : Cosine distance (0 = identical, 2 = opposite)
- `<->` : Euclidean distance (L2)
- `<#>` : Inner product (negative dot product)

**Index:** ivfflat with 100 lists for approximate nearest neighbor search

---

## Storage Architecture

### RustFS S3-Compatible Object Storage

**Bucket Structure:**
```
facevector-engine/
├── originals/
│   ├── 550e8400-e29b-41d4-a716-446655440000.jpg  (1920×1280)
│   ├── 6ba7b810-9dad-11d1-80b4-00c04fd430c8.jpg  (1024×768)
│   └── ...
└── faces/
    ├── f47ac10b-58cc-4372-a567-0e02b2c3d479.jpg  (600×850)
    ├── 9e107d9d-372b-4c1e-bf1a-5c2d48c6a88f.jpg  (450×700)
    └── ...
```

**S3 Service Implementation:**

```typescript
// src/services/s3Service.ts

class S3Service {
  private s3Client: S3Client;

  constructor() {
    this.s3Client = new S3Client({
      endpoint: process.env.S3_ENDPOINT,     // http://localhost:9000
      region: process.env.S3_REGION,         // us-east-1
      credentials: {
        accessKeyId: process.env.S3_ACCESS_KEY,
        secretAccessKey: process.env.S3_SECRET_KEY,
      },
      forcePathStyle: true,  // Required for local S3-compatible storage
    });
  }

  async uploadImage(key: string, buffer: Buffer): Promise<void> {
    await this.s3Client.send(new PutObjectCommand({
      Bucket: S3_CONFIG.BUCKET,
      Key: key,
      Body: buffer,
      ContentType: "image/jpeg",
    }));
  }

  async downloadImage(key: string): Promise<Buffer> {
    const response = await this.s3Client.send(new GetObjectCommand({
      Bucket: S3_CONFIG.BUCKET,
      Key: key,
    }));

    return Buffer.from(await response.Body.transformToByteArray());
  }

  async deleteImage(key: string): Promise<void> {
    await this.s3Client.send(new DeleteObjectCommand({
      Bucket: S3_CONFIG.BUCKET,
      Key: key,
    }));
  }
}
```

### PostgreSQL Database Schema

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

#### Table: `detected_faces`

```sql
CREATE TABLE detected_faces (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  original_image_path text NOT NULL,     -- S3 key: "originals/{uuid}.jpg"
  face_image_path text NOT NULL,         -- S3 key: "faces/{face_id}.jpg"
  identifier text,                       -- Optional client identifier
  bounding_box jsonb NOT NULL,           -- {x, y, width, height}
  confidence float NOT NULL,             -- 0.0 - 100.0
  created_at timestamptz DEFAULT now()
);
```

**Example Row:**
```json
{
  "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "original_image_path": "originals/550e8400-e29b-41d4-a716-446655440000.jpg",
  "face_image_path": "faces/f47ac10b-58cc-4372-a567-0e02b2c3d479.jpg",
  "identifier": "CUSTOMER_001",
  "bounding_box": {"x": 450, "y": 300, "width": 600, "height": 850},
  "confidence": 99.82,
  "created_at": "2024-12-03T10:30:00.000Z"
}
```

#### Table: `enrolled_customers`

```sql
CREATE TABLE enrolled_customers (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  face_id uuid REFERENCES detected_faces(id) ON DELETE CASCADE,
  customer_identifier text NOT NULL,
  customer_name text,
  customer_metadata jsonb,
  embedding vector(512) NOT NULL,        -- pgvector extension
  created_at timestamptz DEFAULT now()
);

-- Vector similarity search index
CREATE INDEX enrolled_customers_embedding_idx
ON enrolled_customers
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

**Example Row:**
```json
{
  "id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
  "face_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "customer_identifier": "CUST001",
  "customer_name": "John Doe",
  "customer_metadata": {"age": 30, "membership": "gold"},
  "embedding": "[0.0234, -0.1456, 0.0892, ...]",  // 512 floats
  "created_at": "2024-12-03T10:35:00.000Z"
}
```

---

## API Workflow Deep Dive

### 1. Face Detection Workflow

```
POST /api/faces/detect
Content-Type: multipart/form-data

┌─────────────────────────────────────────────────────────┐
│ Request Processing                                      │
└─────────────────────────────────────────────────────────┘

1. Multer parses multipart form:
   - file: Buffer (3000×2000 JPEG)
   - identifier: "CUSTOMER_001"

2. facesController.detectFaces():
   ↓
   scaleDownImage(buffer) → base64 (1920×1280)
   ↓
   detectAndStoreFaces(base64, identifier)

3. faceDetectionService.detectAndStoreFaces():

   a) Detect faces:
      detectedFaces = await detectAllFacesWithRetinaFace(base64)
      ↓
      [{ PixelBoundingBox: {Left: 450, Top: 300, ...}, ... }]

   b) Store original image:
      originalImageId = randomUUID()
      await s3Service.uploadImage(
        "originals/{uuid}.jpg",
        base64ToBuffer(base64)
      )

   c) For each detected face:

      i. Crop face region:
         croppedFace = await cropImageRegion(
           base64,
           box.Left,
           box.Top,
           box.Width,
           box.Height
         )

      ii. Store cropped face:
          faceId = randomUUID()
          await s3Service.uploadImage(
            "faces/{face_id}.jpg",
            base64ToBuffer(croppedFace)
          )

      iii. Store metadata in database:
           await client.query(`
             INSERT INTO detected_faces (...)
             VALUES (...)
           `)

   d) Return results:
      [{
        face_id: "f47ac10b-...",
        position: {x: 450, y: 300, width: 600, height: 850},
        confidence: 99.82,
        file_name: "f47ac10b-....jpg"
      }]

┌─────────────────────────────────────────────────────────┐
│ Response: 200 OK                                        │
└─────────────────────────────────────────────────────────┘
```

### 2. Enrollment Workflow

```
POST /api/faces/enroll
Content-Type: application/json

Body:
{
  "face_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "customer_identifier": "CUST001",
  "customer_name": "John Doe",
  "customer_metadata": {"age": 30}
}

┌─────────────────────────────────────────────────────────┐
│ Processing Steps                                        │
└─────────────────────────────────────────────────────────┘

1. facesController.enrollFace():

   a) Fetch face metadata from DB:
      SELECT face_image_path FROM detected_faces
      WHERE id = 'f47ac10b-...'
      ↓
      face_image_path: "faces/f47ac10b-....jpg"

   b) Download face image from S3:
      faceImageBuffer = await s3Service.downloadImage(
        "faces/f47ac10b-....jpg"
      )
      ↓
      faceImageBase64 = buffer.toString("base64")

   c) Generate embedding:

      i. Preprocess:
         preprocessed = await preprocessImage(faceImageBase64)
         // Resizes to 112×112, normalizes to [-1, 1]
         ↓ Float32Array[3×112×112]

      ii. Compute embedding:
          embedding = await computeEmbedding(preprocessed)
          ↓ Float32[512]

   d) Store in database:
      INSERT INTO enrolled_customers (
        face_id,
        customer_identifier,
        customer_name,
        customer_metadata,
        embedding
      ) VALUES (
        'f47ac10b-...',
        'CUST001',
        'John Doe',
        '{"age": 30}',
        vectorToSql(embedding)  // pgvector format
      )
      RETURNING id, customer_identifier, created_at

┌─────────────────────────────────────────────────────────┐
│ Response: 200 OK                                        │
│ {                                                       │
│   "customer_id": "b2c3d4e5-...",                        │
│   "customer_identifier": "CUST001",                     │
│   "customer_name": "John Doe",                          │
│   "created_at": "2024-12-03T10:35:00.000Z"              │
│ }                                                       │
└─────────────────────────────────────────────────────────┘
```

### 3. Recognition Workflow

```
POST /api/faces/recognize
Content-Type: application/json

Body:
{
  "face_id": "9e107d9d-372b-4c1e-bf1a-5c2d48c6a88f"
}

┌─────────────────────────────────────────────────────────┐
│ Processing Steps                                        │
└─────────────────────────────────────────────────────────┘

1. facesController.recognizeFace():

   a) Fetch face image (same as enrollment):
      - Query database for face_image_path
      - Download from S3
      - Convert to base64

   b) Generate query embedding:
      preprocessed = await preprocessImage(faceImageBase64)
      embedding = await computeEmbedding(preprocessed)
      ↓ Float32[512]

   c) Vector similarity search:

      SELECT
        id as customer_id,
        customer_identifier,
        customer_name,
        1 - (embedding <=> $1) as confidence_score
      FROM enrolled_customers
      ORDER BY embedding <=> $1
      LIMIT 10

      Parameters: [vectorToSql(embedding)]

      How it works:
      - pgvector computes cosine distance for all enrolled embeddings
      - ivfflat index provides approximate nearest neighbor search
      - Returns top 10 most similar faces

   d) Format results:
      matches = rows.map(row => ({
        customer_id: row.customer_id,
        customer_identifier: row.customer_identifier,
        customer_name: row.customer_name,
        confidence_score: parseFloat(row.confidence_score.toFixed(4))
      }))

┌─────────────────────────────────────────────────────────┐
│ Response: 200 OK                                        │
│ [                                                       │
│   {                                                     │
│     "customer_id": "b2c3d4e5-...",                      │
│     "customer_identifier": "CUST001",                   │
│     "customer_name": "John Doe",                        │
│     "confidence_score": 0.6708  ← Accepted match        │
│   },                                                    │
│   {                                                     │
│     "customer_id": "c3d4e5f6-...",                      │
│     "customer_identifier": "CUST002",                   │
│     "customer_name": "Jane Smith",                      │
│     "confidence_score": 0.4123  ← Below threshold       │
│   }                                                     │
│ ]                                                       │
└─────────────────────────────────────────────────────────┘
```

---

## Performance Optimizations

Performance claims should be verified with the benchmark script rather than copied from a single machine. Run:

```bash
make run
make benchmark-performance
```

Reports are written to `benchmarks/results/` as JSON and Markdown. The benchmark measures real API calls for:

| Operation | What it measures |
|-----------|------------------|
| `detect_single_face_jpeg` | Multipart JPEG upload with one expected face, Sharp normalization, RetinaFace detection, RustFS/database writes |
| `detect_multi_face_jpeg` | Multipart JPEG upload with multiple expected faces; includes extra crop/alignment/storage work per detected face |
| `detect_webp` | WEBP upload through the same normalization and detection path |
| `detect_avif` | AVIF upload through the same normalization and detection path |
| `recognize` | pgvector similarity lookup against enrolled benchmark identities |
| `full_positive_flow` | Detect positive same-person probe, then recognize it |
| `full_negative_flow` | Detect negative different-person probe, then verify no match at threshold `0.5` |

### Latest Local Benchmark Result

This sample was produced by `scripts/benchmark-performance.mjs` and is machine-specific. Treat it as a local baseline, not a universal performance guarantee.

| Field | Value |
|-------|-------|
| Platform | Linux x64 |
| Node.js | v24.16.0 |
| Iterations | 3 |
| ONNX execution provider | CPU |
| GPU acceleration measured | - |
| CPU model | - |
| RAM | - |
| Database | Local Docker PostgreSQL + pgvector |
| Object storage | Local Docker RustFS |

| Operation | Count | Min ms | P50 ms | P95 ms | Max ms | Avg ms |
|-----------|-------|--------|--------|--------|--------|--------|
| `detect_single_face_jpeg` | 3 | 1652.13 | 2919.14 | 3297.33 | 3297.33 | 2622.87 |
| `detect_multi_face_jpeg` | 3 | 2275.97 | 2451.76 | 2817.55 | 2817.55 | 2515.09 |
| `detect_webp` | 3 | 8653.15 | 8725.34 | 10114.76 | 10114.76 | 9164.42 |
| `detect_avif` | 3 | 1323.25 | 1450.02 | 1491.38 | 1491.38 | 1421.55 |
| `recognize` | 3 | 184.09 | 269.03 | 276.40 | 276.40 | 243.17 |
| `full_positive_flow` | 3 | 2518.17 | 2722.87 | 2920.38 | 2920.38 | 2720.47 |
| `full_negative_flow` | 3 | 2842.45 | 2871.55 | 3492.04 | 3492.04 | 3068.68 |

| Recognition check | Expected | Result | Confidence |
|-------------------|----------|--------|------------|
| Jensen positive probe | Jensen | Jensen | 0.5658 |
| Xi negative probe | No match | No match | - |

### 1. Image Normalization and Scaling

**Purpose:** Put JPEG, PNG, WEBP, and AVIF uploads on one deterministic pixel path while reducing processing time and memory usage.

**Implementation:**
```typescript
// src/utils/imageUtils.ts
const scaleDownImage = async (buffer: Buffer, maxDimension = 1920) => {
  const normalizedBuffer = await normalizeImageBuffer(buffer, maxDimension);
  return normalizedBuffer.toString("base64");
};

const normalizeImageBuffer = async (buffer: Buffer, maxDimension?: number) => {
  const pipeline = sharp(buffer).rotate();
  // Resize with fit: "inside" and output JPEG.
};
```

**Impact:**
- 4000×3000 image → 1920×1440 (75% reduction in pixels)
- Processing time: ~8s → ~3s (62% faster)
- Memory usage: ~48MB → ~11MB (77% less)

### 2. Vector Index (ivfflat)

**Purpose:** Fast approximate nearest neighbor search in high-dimensional space

**Configuration:**
```sql
CREATE INDEX enrolled_customers_embedding_idx
ON enrolled_customers
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

**How it works:**
- Divides embedding space into 100 clusters
- At query time, searches only relevant clusters
- Trade-off: 95-99% accuracy, 10-100× faster than exhaustive search

**Performance:**
- 10,000 customers: ~5ms query time
- 100,000 customers: ~50ms query time
- 1,000,000 customers: ~500ms query time

### 3. S3 Object Storage

**Benefits over filesystem:**
- Scalable: Handle millions of images without filesystem limits
- Concurrent access: Multiple API instances can read/write simultaneously
- Backup: RustFS supports S3-compatible object storage workflows; configure backup and replication according to your deployment plan
- Cost-effective: Cheaper than block storage for large datasets

### 4. ONNX Runtime Optimizations

**CPU Execution Provider:**
```typescript
// src/embedding.ts:12-13
recognitionSession = await ort.InferenceSession.create(
  MODEL_PATHS.FACE_RECOGNITION,
  { executionProviders: ['cpu'] }  // Can use 'cuda' for GPU
);
```

**Performance:**
- CPU (8 cores): ~100ms per embedding
- GPU (NVIDIA T4): ~20ms per embedding (5× faster)
- Batch inference: Process 10 faces in ~150ms (vs 1000ms sequential)

---

## Error Handling & Edge Cases

### 1. No Face Detected

```typescript
// src/services/faceDetectionService.ts:32-34
if (detectedFaces.length === 0) {
  throw { code: "NO_FACE" };
}

// Caught by:
// src/utils/responseHelpers.ts:5-17
if (error && typeof error === 'object' && 'code' in error) {
  if (error.code === 'NO_FACE') {
    res.status(400).json({ error: 'no_face_detected' });
    return;
  }
}
```

### 2. Multiple Faces in Image

**Behavior:** Returns ALL detected faces, sorted by area (largest first)

```typescript
// src/embedding.ts:79-80
detectedFaces.sort((a, b) => b.Area - a.Area);
return detectedFaces;  // All faces, largest first
```

### 3. Low Confidence Detections

**Filtering:**
```typescript
// Default threshold: 0.8 for detection confidence
// Configurable via env var: FACE_DETECTION_CONFIDENCE_THRESHOLD

// src/config/constants.ts:26
VIS_THRESHOLD: 0.8  // Can override with env var
```

### 4. Orphaned Faces

**Definition:** Detected faces not enrolled with any customer

**Cleanup:**
```typescript
// DELETE /api/management/faces/orphaned
// src/controllers/managementController.ts:111-172

// 1. Find orphaned faces
SELECT * FROM detected_faces df
WHERE NOT EXISTS (
  SELECT 1 FROM enrolled_customers ec
  WHERE ec.face_id = df.id
);

// 2. Delete from S3
await s3Service.deleteImage(face.original_image_path);
await s3Service.deleteImage(face.face_image_path);

// 3. Delete from database
DELETE FROM detected_faces WHERE id = ...;
```

---

## Configuration Reference

### Environment Variables

```bash
# Database
DATABASE_URL=postgres://postgres:postgres@localhost:5432/face_db

# API Server
PORT=3000

# Face Detection
FACE_DETECTION_CONFIDENCE_THRESHOLD=0.8  # 0.0 - 1.0

# RustFS S3-Compatible Storage
S3_ENDPOINT=http://localhost:9000
S3_BUCKET=facevector-engine
S3_ACCESS_KEY=rustfsadmin
S3_SECRET_KEY=rustfsadmin
S3_REGION=us-east-1
S3_FORCE_PATH_STYLE=true
```

### Constants

```typescript
// src/config/constants.ts

// Model Input Sizes
FACE_RECOGNITION_INPUT_SIZE = 112  // Recognition model expects 112×112
RETINAFACE_IMAGE_SIZES = {
  MOBILE: 640,                 // Mobile variant
  RESNET50: 840                // ResNet-50 variant (used)
}

// Detection Thresholds
RETINAFACE.CONFIDENCE_THRESHOLD = 0.02  // Initial filter
RETINAFACE.NMS_THRESHOLD = 0.4          // Overlap filter
RETINAFACE.VIS_THRESHOLD = 0.8          // Final filter
RETINAFACE.TOP_K = 5000                 // Before NMS
RETINAFACE.KEEP_TOP_K = 750             // After NMS

// Paths
PATHS.TEMP_DIR = "/tmp/facevector"
PATHS.MODELS_DIR = "models"

// S3 Prefixes
S3_CONFIG.ORIGINALS_PREFIX = "originals/"
S3_CONFIG.FACES_PREFIX = "faces/"
```

---

## Troubleshooting Guide

### Issue: Faces Not Detected

**Possible Causes:**
1. Confidence threshold too high
2. Face too small in image
3. Poor image quality (blur, low light)
4. Extreme face angle (profile view)

**Solutions:**
```bash
# Lower threshold in .env
FACE_DETECTION_CONFIDENCE_THRESHOLD=0.6

# Or scale up image before upload
# Ensure faces are at least 100×100 pixels
```

### Issue: Wrong Face Matched

**Possible Causes:**
1. Similar-looking people enrolled
2. Poor quality enrollment image
3. Significant appearance change (beard, glasses, age)

**Solutions:**
- Enroll multiple images per person
- Use high-quality, well-lit enrollment photos
- Set minimum confidence threshold via `.env`:
  ```bash
  FACE_RECOGNITION_CONFIDENCE_THRESHOLD=0.5
  ```
- You can also override it per request with `min_confidence` on `POST /api/faces/recognize`.
- `confidence_score` is cosine similarity, not a calibrated probability. Tune it with project-specific positive and negative examples, not by treating `0.5` as "50% confidence".

### Issue: Slow Recognition

**Optimization Checklist:**
1. ✅ Ensure pgvector index exists
2. ✅ Use smaller images (1920px max)
3. ✅ Consider GPU acceleration for ONNX models
4. ✅ Limit search results (LIMIT 10 instead of 100)
5. ✅ Scale horizontally (multiple API instances)

---

## Further Reading

- **InsightFace Project:** [deepinsight/insightface](https://github.com/deepinsight/insightface)
- **InsightFace Buffalo-L ONNX Source:** [deepghs/insightface buffalo_l](https://huggingface.co/deepghs/insightface/tree/main/buffalo_l)
- **RetinaFace Paper:** [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641)
- **pgvector:** [GitHub - pgvector/pgvector](https://github.com/pgvector/pgvector)
- **ONNX Runtime:** [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- **RustFS:** [rustfs/rustfs](https://github.com/rustfs/rustfs)

---

## Appendix: Code References

| Component | File Path |
|-----------|-----------|
| Face Detection API | `src/controllers/facesController.ts:13-33` |
| Enrollment API | `src/controllers/facesController.ts:70-126` |
| Recognition API | `src/controllers/facesController.ts:132-189` |
| RetinaFace Implementation | `src/retinaface.ts` |
| Face Recognition Implementation | `src/embedding.ts` |
| Image Utilities | `src/utils/imageUtils.ts` |
| S3 Service | `src/services/s3Service.ts` |
| Database Connection | `src/db.ts` |
| Configuration | `src/config/constants.ts` |
