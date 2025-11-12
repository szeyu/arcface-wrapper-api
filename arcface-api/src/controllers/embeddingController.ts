import { Request, Response } from "express";
import {
  prepareEmbeddingFromPath,
  readImageAsBase64,
  ensureFaceDetected,
  embeddingFromBase64
} from "../services/imageService";
import { insertEmbedding, searchSimilarEmbeddings } from "../services/dbService";

export const storeEmbedding = async (req: Request, res: Response) => {
  try {
    const { image_path } = req.body as { image_path?: string };
    if (!image_path) return res.status(400).json({ error: "Missing image_path" });

    const { embedding, imageBase64 } = await prepareEmbeddingFromPath(image_path);
    const id = await insertEmbedding(embedding, imageBase64);

    res.json({ id });
  } catch (err: any) {
    if (err?.code === "NO_FACE") {
      return res.status(400).json({ error: "no_face_detected" });
    }
    // eslint-disable-next-line no-console
    console.error(err);
    res.status(500).json({ error: "internal error" });
  }
};

export const compareEmbeddings = async (req: Request, res: Response) => {
  try {
    const { image_path_A, image_path_B } = req.body as {
      image_path_A?: string;
      image_path_B?: string;
    };
    if (!image_path_A || !image_path_B) return res.status(400).json({ error: "Missing images" });

    const [preparedA, preparedB] = await Promise.all([
      prepareEmbeddingFromPath(image_path_A),
      prepareEmbeddingFromPath(image_path_B)
    ]);

    const embA = preparedA.embedding;
    const embB = preparedB.embedding;

    // cosine similarity
    const dot = embA.reduce((acc, v, i) => acc + v * embB[i], 0);
    const normA = Math.sqrt(embA.reduce((acc, v) => acc + v * v, 0));
    const normB = Math.sqrt(embB.reduce((acc, v) => acc + v * v, 0));
    const cosine = dot / (normA * normB);

    // euclidean distance
    const euclidean = Math.sqrt(embA.reduce((acc, v, i) => acc + (v - embB[i]) ** 2, 0));

    res.json({ cosine, euclidean });
  } catch (err: any) {
    if (err?.code === "NO_FACE") {
      return res.status(400).json({ error: "no_face_detected" });
    }
    // eslint-disable-next-line no-console
    console.error(err);
    res.status(500).json({ error: "internal error" });
  }
};

export const searchEmbeddings = async (req: Request, res: Response) => {
  try {
    const { image_path, top_k } = req.body as { image_path?: string; top_k?: number | string };
    if (!image_path || top_k === undefined) return res.status(400).json({ error: "Missing params" });

    const limit = Number(top_k);
    if (!Number.isFinite(limit) || limit <= 0) {
      return res.status(400).json({ error: "Invalid top_k" });
    }

    const imageBase64 = await readImageAsBase64(image_path);
    await ensureFaceDetected(imageBase64);
    const embedding = await embeddingFromBase64(imageBase64);

    const result = await searchSimilarEmbeddings(embedding, limit);
    res.json(result);
  } catch (err: any) {
    if (err?.code === "NO_FACE") {
      return res.status(400).json({ error: "no_face_detected" });
    }
    // eslint-disable-next-line no-console
    console.error(err);
    res.status(500).json({ error: "internal error" });
  }
};


