import ort = require("onnxruntime-node");
import { Jimp } from "jimp";

const SCRFD_MODEL = "./models/scrfd.onnx";
const ARC_FACE_MODEL = "./models/arcface.onnx";
const SCRFD_INPUT_SIZE = 640;
const SCRFD_STRIDES = [8, 16, 32];
const SCRFD_ANCHORS_PER_LOCATION = 2;
const SCRFD_CONFIDENCE_THRESHOLD = 0.5;
const SCRFD_MIN_FACE_SIZE = 20;
const SCRFD_MIN_FACE_AREA = SCRFD_MIN_FACE_SIZE * SCRFD_MIN_FACE_SIZE;

let scrfdSession: any;
let arcfaceSession: any;

export const initModels = async () => {
  scrfdSession = await ort.InferenceSession.create(SCRFD_MODEL);
  arcfaceSession = await ort.InferenceSession.create(ARC_FACE_MODEL);
};

// helper: decode base64 to tensor
export const preprocessImage = async (base64: string) => {
  const buffer = Buffer.from(base64, "base64");
  const image = await Jimp.read(buffer);
  await image.resize({ w: 112, h: 112 }); // ArcFace input size

  // Jimp is RGBA; we will extract RGB and normalize per channel
  const data = new Float32Array(3 * 112 * 112);
  let ptr = 0;
  const { width, height, data: bitmapData } = image.bitmap;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = (width * y + x) * 4;
      data[ptr++] = (bitmapData[idx] / 255.0 - 0.5) / 0.5; // R
      data[ptr++] = (bitmapData[idx + 1] / 255.0 - 0.5) / 0.5; // G
      data[ptr++] = (bitmapData[idx + 2] / 255.0 - 0.5) / 0.5; // B
    }
  }
  return data;
};

const preprocessForDetection = async (base64: string) => {
  const buffer = Buffer.from(base64, "base64");
  const image = await Jimp.read(buffer);
  await image.resize({ w: SCRFD_INPUT_SIZE, h: SCRFD_INPUT_SIZE });

  const data = new Float32Array(3 * SCRFD_INPUT_SIZE * SCRFD_INPUT_SIZE);
  let ptr = 0;
  const { width, height, data: bitmapData } = image.bitmap;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = (width * y + x) * 4;
      data[ptr++] = (bitmapData[idx] / 255.0 - 0.5) / 0.5; // R
      data[ptr++] = (bitmapData[idx + 1] / 255.0 - 0.5) / 0.5; // G
      data[ptr++] = (bitmapData[idx + 2] / 255.0 - 0.5) / 0.5; // B
    }
  }
  return data;
};

type OrtTensorLike = {
  dims?: readonly number[];
  data: unknown;
};

const groupScrfdOutputs = (outputs: Record<string, OrtTensorLike>) => {
  const grouped: Record<
    number,
    {
      scores?: Float32Array;
      bboxes?: Float32Array;
    }
  > = {};

  Object.values(outputs).forEach((tensor) => {
    if (!tensor.dims || tensor.dims.length !== 2) return;
    const [numAnchors, channels] = tensor.dims;

    for (const stride of SCRFD_STRIDES) {
      const featSize = SCRFD_INPUT_SIZE / stride;
      const expectedAnchors = featSize * featSize * SCRFD_ANCHORS_PER_LOCATION;
      if (numAnchors !== expectedAnchors) continue;

      grouped[stride] = grouped[stride] || {};
      if (channels === 1) {
        grouped[stride].scores = tensor.data as Float32Array;
      } else if (channels === 4) {
        grouped[stride].bboxes = tensor.data as Float32Array;
      }
      break;
    }
  });

  return grouped;
};

export const detectFace = async (base64: string) => {
  if (!scrfdSession) {
    throw new Error("SCRFD model not initialised");
  }

  const inputData = await preprocessForDetection(base64);
  const tensor = new ort.Tensor("float32", inputData, [1, 3, SCRFD_INPUT_SIZE, SCRFD_INPUT_SIZE]);
  const outputs = await scrfdSession.run({ "input.1": tensor });

  const grouped = groupScrfdOutputs(outputs);

  for (const stride of SCRFD_STRIDES) {
    const entry = grouped[stride];
    if (!entry?.scores || !entry?.bboxes) continue;

    const scores = entry.scores;
    const bboxes = entry.bboxes;

    for (let i = 0; i < scores.length; i += 1) {
      const rawScore = scores[i];
      const prob = 1 / (1 + Math.exp(-rawScore));
      if (prob < SCRFD_CONFIDENCE_THRESHOLD) continue;

      const offset = i * 4;
      // SCRFD bbox format: distances from anchor center (left, top, right, bottom)
      const leftDist = bboxes[offset] * stride;
      const topDist = bboxes[offset + 1] * stride;
      const rightDist = bboxes[offset + 2] * stride;
      const bottomDist = bboxes[offset + 3] * stride;

      // Basic validation: ensure distances are positive and reasonable
      if (leftDist <= 0 || topDist <= 0 || rightDist <= 0 || bottomDist <= 0) {
        continue;
      }

      const width = leftDist + rightDist;
      const height = topDist + bottomDist;

      // Check if face is reasonable size (not too small, not larger than image)
      // Also ensure aspect ratio is reasonable for a face (roughly square to 2:1)
      const aspectRatio = width / height;
      const area = width * height;
      
      // Very lenient size check - just ensure it's not tiny and not larger than image
      // Aspect ratio check helps filter out boxes (which are usually very rectangular)
      // Faces are typically between 0.5 and 2.0 aspect ratio, boxes can be 5:1 or more
      if (
        width >= SCRFD_MIN_FACE_SIZE &&
        height >= SCRFD_MIN_FACE_SIZE &&
        width <= SCRFD_INPUT_SIZE &&
        height <= SCRFD_INPUT_SIZE &&
        aspectRatio >= 0.25 && // Very lenient - allow tall faces
        aspectRatio <= 4.0     // Very lenient - but boxes are often 10:1 or more
      ) {
        return true;
      }
    }
  }
  return false;
};

export const computeEmbedding = async (preprocessed: Float32Array) => {
  const tensor = new ort.Tensor("float32", preprocessed, [1, 3, 112, 112]);
  const results = await arcfaceSession.run({ data: tensor });
  const firstKey = Object.keys(results)[0];
  const embedding = results[firstKey].data as Float32Array;
  return Array.from(embedding);
};


