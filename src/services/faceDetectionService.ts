import { randomUUID } from "crypto";
import { detectAllFacesWithRetinaFace } from "../embedding";
import { alignFaceFromLandmarks, cropImageRegion } from "../utils/imageUtils";
import { client } from "../db";
import { s3Service } from "./s3Service";

interface DetectedFaceResponse {
  face_id: string;
  position: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  confidence: number;
  file_name: string;
}

/**
 * Detect faces in an image, store crops and metadata
 * @param imageBase64 - Base64 encoded image
 * @param identifier - Client-provided identifier for tracking
 * @returns Array of detected faces with metadata
 */
export const detectAndStoreFaces = async (
  imageBase64: string,
  identifier?: string
): Promise<DetectedFaceResponse[]> => {
  // Detect all faces in the image
  const detectedFaces = await detectAllFacesWithRetinaFace(imageBase64);

  if (detectedFaces.length === 0) {
    throw { code: "NO_FACE" };
  }

  const results: DetectedFaceResponse[] = [];

  // Store original image to S3
  const originalImageId = randomUUID();
  const originalImageKey = `originals/${originalImageId}.jpg`;
  const originalImageBuffer = Buffer.from(imageBase64, "base64");
  await s3Service.uploadImage(originalImageKey, originalImageBuffer);

  // Process each detected face
  for (const face of detectedFaces) {
    const faceId = randomUUID();
    const pixelBox = face.PixelBoundingBox;
    const confidence = face.Confidence;

    // The recognition model expects a landmark-aligned 112x112 face, not a raw bounding-box crop.
    const croppedFaceBase64 = face.Landmarks?.length
      ? await alignFaceFromLandmarks(imageBase64, face.Landmarks)
      : await cropImageRegion(
          imageBase64,
          pixelBox.Left,
          pixelBox.Top,
          pixelBox.Width,
          pixelBox.Height
        );

    // Store cropped face image to S3
    const faceImageKey = `faces/${faceId}.jpg`;
    const faceImageBuffer = Buffer.from(croppedFaceBase64, "base64");
    await s3Service.uploadImage(faceImageKey, faceImageBuffer);

    // Store metadata in database
    const boundingBox = {
      x: pixelBox.Left,
      y: pixelBox.Top,
      width: pixelBox.Width,
      height: pixelBox.Height,
    };

    await client.query(
      `INSERT INTO detected_faces
       (id, original_image_path, face_image_path, identifier, bounding_box, confidence)
       VALUES ($1, $2, $3, $4, $5, $6)`,
      [
        faceId,
        originalImageKey,
        faceImageKey,
        identifier || null,
        JSON.stringify(boundingBox),
        confidence,
      ]
    );

    results.push({
      face_id: faceId,
      position: boundingBox,
      confidence: confidence,
      file_name: `${faceId}.jpg`,
    });
  }

  return results;
};
