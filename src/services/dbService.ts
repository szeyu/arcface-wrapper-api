import { client, vectorToSql } from "../db";

export const insertEmbedding = async (
  embedding: number[],
  imagePath: string,
  customId?: string
): Promise<string> => {
  let result;
  if (customId) {
    result = await client.query(
      "INSERT INTO face_embeddings (id, embedding, image_path) VALUES ($1, $2, $3) RETURNING id",
      [customId, vectorToSql(embedding), imagePath]
    );
  } else {
    result = await client.query(
      "INSERT INTO face_embeddings (embedding, image_path) VALUES ($1, $2) RETURNING id",
      [vectorToSql(embedding), imagePath]
    );
  }
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
  const result = await client.query("SELECT image_path FROM face_embeddings WHERE id=$1", [id]);
  if (!result.rows.length) return null;
  return result.rows[0].image_path as string;
};

export const deleteEmbeddingById = async (id: string): Promise<boolean> => {
  const result = await client.query("DELETE FROM face_embeddings WHERE id=$1 RETURNING id", [id]);
  return result.rows.length > 0;
};

export const searchSimilarEmbeddings = async (embedding: number[], limit: number) => {
  const result = await client.query(
    `WITH scored_matches AS MATERIALIZED (
       SELECT id, 1 - (embedding <=> $1) as cosine
       FROM face_embeddings
     )
     SELECT id, cosine
     FROM scored_matches
     ORDER BY cosine DESC
     LIMIT $2`,
    [vectorToSql(embedding), limit]
  );
  return result.rows;
};

