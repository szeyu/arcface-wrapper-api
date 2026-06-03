import { promises as fs } from "fs";
import path from "path";
import { detectAllFacesWithRetinaFace } from "../embedding";
import { alignFaceFromLandmarks, cropImageRegion } from "../utils/imageUtils";
import { NoFaceDetectedError } from "../utils/errors";

// Use /tmp inside Docker for ephemeral temporary files
const CROPPED_FACES_DIR = "/tmp/facevector/cropped_faces";

/**
 * Clear all files in the cropped_faces directory
 */
export const clearCroppedFacesDir = async (): Promise<void> => {
  try {
    // Check if directory exists
    await fs.access(CROPPED_FACES_DIR);

    // Read all files in directory
    const files = await fs.readdir(CROPPED_FACES_DIR);

    // Delete each file
    await Promise.all(
      files.map(file => fs.unlink(path.join(CROPPED_FACES_DIR, file)))
    );
  } catch (err: unknown) {
    // If directory doesn't exist, create it
    if (err && typeof err === "object" && "code" in err && err.code === "ENOENT") {
      await fs.mkdir(CROPPED_FACES_DIR, { recursive: true });
    } else {
      throw err;
    }
  }
};

/**
 * Detect all faces in an image, crop them, and save to cropped_faces directory
 * Returns the number of faces detected and cropped
 */
export const cropAndSaveFaces = async (imageBase64: string): Promise<number> => {
  // Detect all faces using RetinaFace
  const faces = await detectAllFacesWithRetinaFace(imageBase64);

  if (faces.length === 0) {
    throw new NoFaceDetectedError();
  }

  // Ensure cropped_faces directory exists and is empty
  await clearCroppedFacesDir();

  // Crop and save each face
  for (let i = 0; i < faces.length; i++) {
    const face = faces[i];
    const bbox = face.PixelBoundingBox;

    const croppedBase64 = face.Landmarks?.length
      ? await alignFaceFromLandmarks(imageBase64, face.Landmarks)
      : await cropImageRegion(
          imageBase64,
          bbox.Left,
          bbox.Top,
          bbox.Width,
          bbox.Height
        );

    // Save cropped face to file
    const filename = `face_${i}.jpg`;
    const filepath = path.join(CROPPED_FACES_DIR, filename);
    const buffer = Buffer.from(croppedBase64, "base64");
    await fs.writeFile(filepath, buffer);
  }

  return faces.length;
};

/**
 * Get all cropped face file paths from the cropped_faces directory
 */
export const getCroppedFacePaths = async (): Promise<string[]> => {
  try {
    const files = await fs.readdir(CROPPED_FACES_DIR);
    return files
      .filter(file => file.startsWith("face_") && file.endsWith(".jpg"))
      .sort() // Sort to maintain order (face_0, face_1, etc.)
      .map(file => path.join(CROPPED_FACES_DIR, file));
  } catch (err: unknown) {
    if (err && typeof err === "object" && "code" in err && err.code === "ENOENT") {
      return [];
    }
    throw err;
  }
};
