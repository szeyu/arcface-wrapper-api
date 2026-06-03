import { Jimp } from "jimp";
import sharp from "sharp";
import { FACE_RECOGNITION_INPUT_SIZE } from "../config/constants";
import type { Landmark } from "../types/face";

const NORMALIZED_JPEG_QUALITY = 92;

/**
 * Normalize supported input images into auto-rotated JPEG buffers.
 * This keeps JPEG, PNG, WEBP, and AVIF uploads on one deterministic path before
 * the rest of the pipeline reads pixels with Jimp.
 */
export const normalizeImageBuffer = async (
  buffer: Buffer,
  maxDimension?: number
): Promise<Buffer> => {
  const pipeline = sharp(buffer, { failOn: "error" }).rotate();

  if (maxDimension) {
    pipeline.resize({
      width: maxDimension,
      height: maxDimension,
      fit: "inside",
      withoutEnlargement: true,
    });
  }

  return pipeline
    .jpeg({ quality: NORMALIZED_JPEG_QUALITY, mozjpeg: true })
    .toBuffer();
};

/**
 * Convert base64 encoded image to Jimp instance
 * This is a common operation used throughout the codebase
 */
export const base64ToJimp = async (base64: string): Promise<Awaited<ReturnType<typeof Jimp.read>>> => {
  const buffer = Buffer.from(base64, "base64");
  const normalizedBuffer = await normalizeImageBuffer(buffer);
  return await Jimp.read(normalizedBuffer);
};

/**
 * Convert Buffer to Jimp instance
 * Used for processing uploaded files from multer
 */
export const bufferToJimp = async (buffer: Buffer): Promise<Awaited<ReturnType<typeof Jimp.read>>> => {
  const normalizedBuffer = await normalizeImageBuffer(buffer);
  return await Jimp.read(normalizedBuffer);
};

/**
 * Convert Jimp instance to base64 string
 */
export const jimpToBase64 = async (image: Awaited<ReturnType<typeof Jimp.read>>): Promise<string> => {
  const buffer = await image.getBuffer("image/jpeg");
  return buffer.toString("base64");
};

/**
 * Get image dimensions from base64 encoded image
 */
export const getImageDimensions = async (
  base64: string
): Promise<{ width: number; height: number }> => {
  const image = await base64ToJimp(base64);
  return {
    width: image.bitmap.width,
    height: image.bitmap.height,
  };
};

/**
 * Crop a region from an image and return as base64
 */
export const cropImageRegion = async (
  base64: string,
  x: number,
  y: number,
  width: number,
  height: number
): Promise<string> => {
  const image = await base64ToJimp(base64);

  // Ensure coordinates are within image bounds
  const clampedX = Math.max(0, Math.min(x, image.bitmap.width - 1));
  const clampedY = Math.max(0, Math.min(y, image.bitmap.height - 1));
  const clampedW = Math.min(width, image.bitmap.width - clampedX);
  const clampedH = Math.min(height, image.bitmap.height - clampedY);

  // Crop the image
  image.crop({ x: clampedX, y: clampedY, w: clampedW, h: clampedH });

  // Convert to base64
  const croppedBuffer = await image.getBuffer("image/jpeg");
  return croppedBuffer.toString("base64");
};

const FACE_RECOGNITION_TEMPLATE: Array<[number, number]> = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];

const LANDMARK_ORDER = ["eyeLeft", "eyeRight", "nose", "mouthLeft", "mouthRight"];

const solve3x3 = (
  matrix: [[number, number, number], [number, number, number], [number, number, number]],
  vector: [number, number, number]
): [number, number, number] => {
  const [[a, b, c], [d, e, f], [g, h, i]] = matrix;
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);

  if (Math.abs(det) < 1e-8) {
    throw new Error("Cannot solve face alignment transform");
  }

  const inv: [[number, number, number], [number, number, number], [number, number, number]] = [
    [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
    [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
    [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
  ];

  return [
    inv[0][0] * vector[0] + inv[0][1] * vector[1] + inv[0][2] * vector[2],
    inv[1][0] * vector[0] + inv[1][1] * vector[1] + inv[1][2] * vector[2],
    inv[2][0] * vector[0] + inv[2][1] * vector[1] + inv[2][2] * vector[2],
  ];
};

const estimateAffine = (
  source: Array<[number, number]>,
  target: Array<[number, number]>
): [number, number, number, number, number, number] => {
  const normal: [[number, number, number], [number, number, number], [number, number, number]] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const rhsX: [number, number, number] = [0, 0, 0];
  const rhsY: [number, number, number] = [0, 0, 0];

  for (let idx = 0; idx < source.length; idx += 1) {
    const [u, v] = source[idx];
    const [x, y] = target[idx];
    const row: [number, number, number] = [u, v, 1];

    for (let r = 0; r < 3; r += 1) {
      rhsX[r] += row[r] * x;
      rhsY[r] += row[r] * y;

      for (let c = 0; c < 3; c += 1) {
        normal[r][c] += row[r] * row[c];
      }
    }
  }

  const [a, b, c] = solve3x3(normal, rhsX);
  const [d, e, f] = solve3x3(normal, rhsY);
  return [a, b, c, d, e, f];
};

const getOrderedLandmarkPoints = (landmarks: Landmark[]): Array<[number, number]> => {
  return LANDMARK_ORDER.map((type) => {
    const landmark = landmarks.find((item) => item.Type === type);
    if (!landmark) {
      throw new Error(`Missing ${type} landmark for face alignment`);
    }
    return [landmark.PixelX, landmark.PixelY];
  });
};

const sampleBilinear = (
  sourceData: Buffer | Uint8Array,
  width: number,
  height: number,
  x: number,
  y: number
): [number, number, number, number] => {
  const clampedX = Math.max(0, Math.min(width - 1, x));
  const clampedY = Math.max(0, Math.min(height - 1, y));
  const x0 = Math.floor(clampedX);
  const y0 = Math.floor(clampedY);
  const x1 = Math.min(width - 1, x0 + 1);
  const y1 = Math.min(height - 1, y0 + 1);
  const dx = clampedX - x0;
  const dy = clampedY - y0;

  const read = (px: number, py: number): [number, number, number, number] => {
    const offset = (py * width + px) * 4;
    return [
      sourceData[offset],
      sourceData[offset + 1],
      sourceData[offset + 2],
      sourceData[offset + 3],
    ];
  };

  const topLeft = read(x0, y0);
  const topRight = read(x1, y0);
  const bottomLeft = read(x0, y1);
  const bottomRight = read(x1, y1);

  return [0, 1, 2, 3].map((channel) => {
    const top = topLeft[channel] * (1 - dx) + topRight[channel] * dx;
    const bottom = bottomLeft[channel] * (1 - dx) + bottomRight[channel] * dx;
    return Math.round(top * (1 - dy) + bottom * dy);
  }) as [number, number, number, number];
};

/**
 * Align a detected face to the InsightFace 112x112 landmark template.
 */
export const alignFaceFromLandmarks = async (
  base64: string,
  landmarks: Landmark[]
): Promise<string> => {
  const image = await base64ToJimp(base64);
  const output = new Jimp({
    width: FACE_RECOGNITION_INPUT_SIZE,
    height: FACE_RECOGNITION_INPUT_SIZE,
    color: 0x000000ff,
  });

  const detectedPoints = getOrderedLandmarkPoints(landmarks);
  const [a, b, c, d, e, f] = estimateAffine(FACE_RECOGNITION_TEMPLATE, detectedPoints);

  for (let y = 0; y < FACE_RECOGNITION_INPUT_SIZE; y += 1) {
    for (let x = 0; x < FACE_RECOGNITION_INPUT_SIZE; x += 1) {
      const srcX = a * x + b * y + c;
      const srcY = d * x + e * y + f;
      const [r, g, blue, alpha] = sampleBilinear(
        image.bitmap.data,
        image.bitmap.width,
        image.bitmap.height,
        srcX,
        srcY
      );
      const offset = (y * FACE_RECOGNITION_INPUT_SIZE + x) * 4;
      output.bitmap.data[offset] = r;
      output.bitmap.data[offset + 1] = g;
      output.bitmap.data[offset + 2] = blue;
      output.bitmap.data[offset + 3] = alpha;
    }
  }

  const buffer = await output.getBuffer("image/jpeg");
  return buffer.toString("base64");
};

/**
 * Scale down image if it exceeds maximum dimension
 * Maintains aspect ratio and improves processing performance
 * @param buffer - Image buffer from file upload
 * @param maxDimension - Maximum width or height (default: 1920)
 * @returns Scaled image as base64 string
 */
export const scaleDownImage = async (
  buffer: Buffer,
  maxDimension: number = 1920
): Promise<string> => {
  const normalizedBuffer = await normalizeImageBuffer(buffer, maxDimension);
  return normalizedBuffer.toString("base64");
};
