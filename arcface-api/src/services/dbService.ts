import { client, vectorToSql } from "../db";

export const insertEmbedding = async (embedding: number[], imageBase64: string): Promise<string> => {
  const result = await client.query(
    "INSERT INTO face_embeddings (embedding, image_base64) VALUES ($1, $2) RETURNING id",
    [vectorToSql(embedding), imageBase64]
  );
  return result.rows[0].id as string;
};

export const listEmbeddings = async (limit: number) => {
  const result = await client.query(
    "SELECT id, created_at FROM face_embeddings ORDER BY created_at DESC LIMIT $1",
    [limit]
  );
  return result.rows;
};

export const getImageById = async (id: string): Promise<string | null> => {
  const result = await client.query("SELECT image_base64 FROM face_embeddings WHERE id=$1", [id]);
  if (!result.rows.length) return null;
  return result.rows[0].image_base64 as string;
};

export const deleteEmbeddingById = async (id: string): Promise<boolean> => {
  const result = await client.query("DELETE FROM face_embeddings WHERE id=$1 RETURNING id", [id]);
  return result.rows.length > 0;
};

export const searchSimilarEmbeddings = async (embedding: number[], limit: number) => {
  const result = await client.query(
    "SELECT id, 1 - (embedding <=> $1) as cosine FROM face_embeddings ORDER BY embedding <-> $1 LIMIT $2",
    [vectorToSql(embedding), limit]
  );
  return result.rows;
};


