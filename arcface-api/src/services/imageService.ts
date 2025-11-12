import { promises as fs } from "fs";
import path from "path";
import { preprocessImage, computeEmbedding, detectFace } from "../embedding";

const resolvePath = (imagePath: string) =>
  path.isAbsolute(imagePath) ? imagePath : path.resolve(process.cwd(), imagePath);

export const readImageAsBase64 = async (imagePath: string): Promise<string> => {
  const absolutePath = resolvePath(imagePath);
  const fileBuffer = await fs.readFile(absolutePath);
  return fileBuffer.toString("base64");
};

export const ensureFaceDetected = async (imageBase64: string): Promise<void> => {
  const hasFace = await detectFace(imageBase64);
  if (!hasFace) {
    const error = new Error("no_face_detected");
    // @ts-ignore add custom property for easier handling
    error.code = "NO_FACE";
    throw error;
  }
};

export const embeddingFromBase64 = async (imageBase64: string): Promise<number[]> => {
  const tensor = await preprocessImage(imageBase64);
  return computeEmbedding(tensor);
};

export const prepareEmbeddingFromPath = async (imagePath: string) => {
  const imageBase64 = await readImageAsBase64(imagePath);
  await ensureFaceDetected(imageBase64);
  const embedding = await embeddingFromBase64(imageBase64);
  return { imageBase64, embedding };
};


