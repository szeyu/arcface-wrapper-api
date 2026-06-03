import ort = require("onnxruntime-node");
import { FACE_RECOGNITION_INPUT_SIZE, MODEL_PATHS } from "./config/constants";
import type { DetectedFace } from "./types/face";
import { base64ToJimp } from "./utils/imageUtils";

// Re-export types for backward compatibility
export type { DetectedFace, Landmark } from "./types/face";

type InferenceSession = Awaited<ReturnType<typeof ort.InferenceSession.create>>;
let recognitionSession: InferenceSession | null = null;

export const initModels = async () => {
  recognitionSession = await ort.InferenceSession.create(MODEL_PATHS.FACE_RECOGNITION);

  // Load RetinaFace model (required)
  const { initRetinaFaceModel } = await import("./retinaface.js");
  await initRetinaFaceModel(MODEL_PATHS.RETINAFACE, "resnet50");
};

// helper: decode base64 to tensor
export const preprocessImage = async (base64: string) => {
  const image = await base64ToJimp(base64);
  await image.resize({ w: FACE_RECOGNITION_INPUT_SIZE, h: FACE_RECOGNITION_INPUT_SIZE });

  // Jimp is RGBA; the InsightFace export expects OpenCV-style BGR in NCHW layout.
  const data = new Float32Array(3 * FACE_RECOGNITION_INPUT_SIZE * FACE_RECOGNITION_INPUT_SIZE);
  const { width, height, data: bitmapData } = image.bitmap;
  const imageSize = width * height;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = (width * y + x) * 4;
      const pixelIndex = width * y + x;

      data[pixelIndex] = (bitmapData[idx + 2] / 255.0 - 0.5) / 0.5; // B
      data[imageSize + pixelIndex] = (bitmapData[idx + 1] / 255.0 - 0.5) / 0.5; // G
      data[2 * imageSize + pixelIndex] = (bitmapData[idx] / 255.0 - 0.5) / 0.5; // R
    }
  }
  return data;
};

export const computeEmbedding = async (preprocessed: Float32Array) => {
  if (!recognitionSession) {
    throw new Error("Face recognition model not initialized");
  }
  const tensor = new ort.Tensor("float32", preprocessed, [1, 3, FACE_RECOGNITION_INPUT_SIZE, FACE_RECOGNITION_INPUT_SIZE]);
  const inputName = recognitionSession.inputNames[0];
  const outputName = recognitionSession.outputNames[0];
  const results = await recognitionSession.run({ [inputName]: tensor });
  const firstKey = outputName || Object.keys(results)[0];
  const embedding = results[firstKey].data as Float32Array;
  return Array.from(embedding);
};

/**
 * Detect all faces using RetinaFace
 * Returns faces with same interface as detectAllFaces
 */
export const detectAllFacesWithRetinaFace = async (base64: string, visThreshold: number = 0.6): Promise<DetectedFace[]> => {
  const {
    isRetinaFaceModelAvailable,
    detectFacesRetinaFace,
    convertRetinaFaceToDetectedFace
  } = await import("./retinaface.js");

  if (!isRetinaFaceModelAvailable()) {
    throw new Error("RetinaFace model not initialized");
  }

  // Get original image dimensions
  const originalImage = await base64ToJimp(base64);
  const originalWidth = originalImage.bitmap.width;
  const originalHeight = originalImage.bitmap.height;

  // Detect faces using RetinaFace
  const detections = await detectFacesRetinaFace(base64, visThreshold);

  // Convert to standard DetectedFace format
  const detectedFaces = detections.map((detection: { bbox: number[]; confidence: number; landmarks: number[][] }) =>
    convertRetinaFaceToDetectedFace(detection, originalWidth, originalHeight)
  );

  // Sort by area (largest first)
  detectedFaces.sort((a: DetectedFace, b: DetectedFace) => b.Area - a.Area);

  return detectedFaces;
};

/**
 * Compare two face embeddings and return similarity metrics
 */
export const compareEmbeddings = (
  embedding1: number[],
  embedding2: number[]
): { cosineSimilarity: number; euclideanDistance: number } => {
  if (embedding1.length !== embedding2.length) {
    throw new Error('Embeddings must have the same dimension');
  }

  // Calculate cosine similarity
  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;

  for (let i = 0; i < embedding1.length; i++) {
    dotProduct += embedding1[i] * embedding2[i];
    norm1 += embedding1[i] * embedding1[i];
    norm2 += embedding2[i] * embedding2[i];
  }

  const cosineSimilarity = dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));

  // Calculate Euclidean distance
  let sumSquares = 0;
  for (let i = 0; i < embedding1.length; i++) {
    const diff = embedding1[i] - embedding2[i];
    sumSquares += diff * diff;
  }
  const euclideanDistance = Math.sqrt(sumSquares);

  return {
    cosineSimilarity,
    euclideanDistance
  };
};
