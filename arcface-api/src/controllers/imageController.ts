import { Request, Response } from "express";
import { listEmbeddings, getImageById, deleteEmbeddingById } from "../services/dbService";
import { saveImageToOutput } from "../services/outputService";

export const listImages = async (req: Request, res: Response) => {
  const limitValue = parseInt((req.query.limit as string) ?? "10", 10);
  const limit = Number.isNaN(limitValue) ? 10 : limitValue;
  const rows = await listEmbeddings(limit);
  res.json(rows);
};

export const getImage = async (req: Request, res: Response) => {
  try {
    const { id } = req.params as { id: string };
    const imageBase64 = await getImageById(id);
    if (!imageBase64) return res.status(404).json({ error: "not found" });

    const buffer = Buffer.from(imageBase64, "base64");
    const savedPath = await saveImageToOutput(id, buffer);

    res.json({
      image_base64: imageBase64,
      saved_to: savedPath
    });
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error(err);
    res.status(500).json({ error: "internal error" });
  }
};

export const deleteImage = async (req: Request, res: Response) => {
  const { id } = req.params as { id: string };
  const deleted = await deleteEmbeddingById(id);
  if (!deleted) return res.status(404).json({ error: "not found" });
  res.json({ deleted_id: id });
};


