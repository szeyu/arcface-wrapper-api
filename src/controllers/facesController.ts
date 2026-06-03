import { Request, Response } from "express";
import { detectAndStoreFaces } from "../services/faceDetectionService";
import { scaleDownImage } from "../utils/imageUtils";
import { sendErrorResponse } from "../utils/responseHelpers";
import { client, vectorToSql } from "../db";
import { computeEmbedding, preprocessImage } from "../embedding";
import { s3Service } from "../services/s3Service";
import { getDefaultRecognitionConfidenceThreshold } from "../config/constants";

/**
 * POST /faces/detect
 * Detect faces in uploaded image, store crops and metadata
 */
export const detectFaces = async (req: Request, res: Response): Promise<void> => {
  try {
    if (!req.file) {
      res.status(400).json({ error: "No image file provided" });
      return;
    }

    // Get optional identifier from form data
    const identifier = req.body.identifier as string | undefined;

    // Scale down image for better performance
    const imageBase64 = await scaleDownImage(req.file.buffer);

    // Detect faces and store metadata
    const detectedFaces = await detectAndStoreFaces(imageBase64, identifier);

    res.json(detectedFaces);
  } catch (error: unknown) {
    sendErrorResponse(res, error);
  }
};

/**
 * GET /faces/:face_id
 * Retrieve stored face image by face_id
 */
export const getFaceImage = async (req: Request, res: Response): Promise<void> => {
  try {
    const { face_id } = req.params;

    // Fetch face metadata from database
    const result = await client.query(
      "SELECT face_image_path FROM detected_faces WHERE id = $1",
      [face_id]
    );

    if (result.rows.length === 0) {
      res.status(404).json({ error: "Face not found" });
      return;
    }

    const faceImageKey = result.rows[0].face_image_path;

    // Download and return the face image from S3
    const imageBuffer = await s3Service.downloadImage(faceImageKey);
    res.set("Content-Type", "image/jpeg");
    res.send(imageBuffer);
  } catch (error: unknown) {
    console.error("Error in getFaceImage:", error);
    sendErrorResponse(res, error);
  }
};

/**
 * POST /faces/enroll
 * Enroll a customer using a detected face_id
 */
export const enrollFace = async (req: Request, res: Response): Promise<void> => {
  try {
    const { face_id, customer_identifier, customer_name, customer_metadata } = req.body;

    if (!face_id || !customer_identifier) {
      res.status(400).json({ error: "face_id and customer_identifier are required" });
      return;
    }

    // Fetch face image from database
    const faceResult = await client.query(
      "SELECT face_image_path FROM detected_faces WHERE id = $1",
      [face_id]
    );

    if (faceResult.rows.length === 0) {
      res.status(404).json({ error: "Face not found" });
      return;
    }

    const faceImageKey = faceResult.rows[0].face_image_path;

    // Download face image from S3 and convert to base64
    const faceImageBuffer = await s3Service.downloadImage(faceImageKey);
    const faceImageBase64 = faceImageBuffer.toString("base64");

    // Generate embedding
    const preprocessed = await preprocessImage(faceImageBase64);
    const embedding = await computeEmbedding(preprocessed);
    const embeddingArray = Array.from(embedding);

    // Store customer record with embedding
    const insertResult = await client.query(
      `INSERT INTO enrolled_customers
       (face_id, customer_identifier, customer_name, customer_metadata, embedding)
       VALUES ($1, $2, $3, $4, $5)
       RETURNING id, customer_identifier, customer_name, created_at`,
      [
        face_id,
        customer_identifier,
        customer_name || null,
        customer_metadata ? JSON.stringify(customer_metadata) : null,
        vectorToSql(embeddingArray),
      ]
    );

    res.json({
      customer_id: insertResult.rows[0].id,
      customer_identifier: insertResult.rows[0].customer_identifier,
      customer_name: insertResult.rows[0].customer_name,
      created_at: insertResult.rows[0].created_at,
    });
  } catch (error: unknown) {
    console.error("Error in enrollFace:", error);
    sendErrorResponse(res, error);
  }
};

/**
 * POST /faces/recognize
 * Recognize a face by searching for similar enrolled customers
 */
export const recognizeFace = async (req: Request, res: Response): Promise<void> => {
  try {
    const { face_id, min_confidence } = req.body;

    if (!face_id) {
      res.status(400).json({ error: "face_id is required" });
      return;
    }

    const recognitionThreshold =
      typeof min_confidence === "number"
        ? min_confidence
        : getDefaultRecognitionConfidenceThreshold();

    if (
      !Number.isFinite(recognitionThreshold) ||
      recognitionThreshold < 0 ||
      recognitionThreshold > 1
    ) {
      res.status(400).json({ error: "min_confidence must be a number between 0 and 1" });
      return;
    }

    // Fetch face image from database
    const faceResult = await client.query(
      "SELECT face_image_path FROM detected_faces WHERE id = $1",
      [face_id]
    );

    if (faceResult.rows.length === 0) {
      res.status(404).json({ error: "Face not found" });
      return;
    }

    const faceImageKey = faceResult.rows[0].face_image_path;

    // Download face image from S3 and convert to base64
    const faceImageBuffer = await s3Service.downloadImage(faceImageKey);
    const faceImageBase64 = faceImageBuffer.toString("base64");

    // Generate embedding for the query face
    const preprocessed = await preprocessImage(faceImageBase64);
    const embedding = await computeEmbedding(preprocessed);
    const embeddingArray = Array.from(embedding);

    // Search for similar customers in vector DB
    const searchResult = await client.query(
      `WITH scored_matches AS MATERIALIZED (
         SELECT
           id as customer_id,
           customer_identifier,
           customer_name,
           1 - (embedding <=> $1) as confidence_score
         FROM enrolled_customers
       )
       SELECT
         customer_id,
         customer_identifier,
         customer_name,
         confidence_score
       FROM scored_matches
       ORDER BY confidence_score DESC
       LIMIT 10`,
      [vectorToSql(embeddingArray)]
    );

    // Format results
    const matches = searchResult.rows
      .map((row) => ({
        customer_id: row.customer_id,
        customer_identifier: row.customer_identifier,
        customer_name: row.customer_name,
        confidence_score: parseFloat(row.confidence_score.toFixed(4)),
      }))
      .filter((match) => match.confidence_score >= recognitionThreshold);

    res.json(matches);
  } catch (error: unknown) {
    console.error("Error in recognizeFace:", error);
    sendErrorResponse(res, error);
  }
};
