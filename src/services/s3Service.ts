import {
  S3Client,
  GetObjectCommand,
  DeleteObjectCommand,
  HeadBucketCommand,
  CreateBucketCommand,
} from "@aws-sdk/client-s3";
import { Upload } from "@aws-sdk/lib-storage";
import { Readable } from "stream";

const isNoSuchKeyError = (error: unknown): boolean => {
  if (!error || typeof error !== "object") {
    return false;
  }

  const maybeError = error as { name?: string; Code?: string };
  return maybeError.name === "NoSuchKey" || maybeError.Code === "NoSuchKey";
};

/**
 * S3-compatible storage service for RustFS.
 * Handles all object storage operations for face images.
 */
class S3Service {
  private client: S3Client;
  private bucket: string;

  constructor() {
    const endpoint = process.env.S3_ENDPOINT || "http://localhost:9000";
    const region = process.env.S3_REGION || "us-east-1";
    const accessKeyId = process.env.S3_ACCESS_KEY || "rustfsadmin";
    const secretAccessKey = process.env.S3_SECRET_KEY || "rustfsadmin";
    const forcePathStyle = process.env.S3_FORCE_PATH_STYLE !== "false";

    this.client = new S3Client({
      endpoint,
      region,
      credentials: {
        accessKeyId,
        secretAccessKey,
      },
      forcePathStyle,
    });

    this.bucket = process.env.S3_BUCKET || "facevector-engine";
  }

  /**
   * Upload an image to S3
   * @param key - S3 object key (e.g., "originals/uuid.jpg")
   * @param buffer - Image buffer
   * @param contentType - Content type (default: image/jpeg)
   * @returns S3 object key
   */
  async uploadImage(
    key: string,
    buffer: Buffer,
    contentType: string = "image/jpeg"
  ): Promise<string> {
    try {
      const upload = new Upload({
        client: this.client,
        params: {
          Bucket: this.bucket,
          Key: key,
          Body: buffer,
          ContentType: contentType,
        },
      });

      await upload.done();
      return key;
    } catch (error) {
      console.error(`Failed to upload image to S3: ${key}`, error);
      throw new Error(`S3 upload failed: ${error}`);
    }
  }

  /**
   * Download an image from S3
   * @param key - S3 object key
   * @returns Image buffer
   */
  async downloadImage(key: string): Promise<Buffer> {
    try {
      const command = new GetObjectCommand({
        Bucket: this.bucket,
        Key: key,
      });

      const response = await this.client.send(command);

      if (!response.Body) {
        throw new Error("Empty response body from S3");
      }

      // Convert stream to buffer
      const stream = response.Body as Readable;
      const chunks: Buffer[] = [];

      return new Promise((resolve, reject) => {
        stream.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
        stream.on("error", reject);
        stream.on("end", () => resolve(Buffer.concat(chunks)));
      });
    } catch (error) {
      if (!isNoSuchKeyError(error)) {
        console.error(`Failed to download image from S3: ${key}`, error);
      }
      throw new Error(`S3 download failed: ${error}`);
    }
  }

  /**
   * Delete an image from S3
   * @param key - S3 object key
   */
  async deleteImage(key: string): Promise<void> {
    try {
      const command = new DeleteObjectCommand({
        Bucket: this.bucket,
        Key: key,
      });

      await this.client.send(command);
    } catch (error) {
      console.error(`Failed to delete image from S3: ${key}`, error);
      throw new Error(`S3 delete failed: ${error}`);
    }
  }

  /**
   * Check if bucket exists, create if it doesn't
   */
  async ensureBucketExists(): Promise<void> {
    try {
      // Try to access the bucket
      const headCommand = new HeadBucketCommand({
        Bucket: this.bucket,
      });

      await this.client.send(headCommand);
      console.log(`✓ S3 bucket '${this.bucket}' exists`);
    } catch (error: unknown) {
      // Bucket doesn't exist, create it
      const awsError = error as { name?: string; $metadata?: { httpStatusCode?: number } };
      if (awsError.name === "NotFound" || awsError.$metadata?.httpStatusCode === 404) {
        try {
          const createCommand = new CreateBucketCommand({
            Bucket: this.bucket,
          });

          await this.client.send(createCommand);
          console.log(`✓ Created S3 bucket '${this.bucket}'`);
        } catch (createError) {
          console.error(`Failed to create S3 bucket '${this.bucket}'`, createError);
          throw new Error(`Failed to create S3 bucket: ${createError}`);
        }
      } else {
        console.error(`Failed to check S3 bucket '${this.bucket}'`, error);
        throw new Error(`Failed to check S3 bucket: ${error}`);
      }
    }
  }

  /**
   * Get the S3 client instance
   * Useful for advanced operations
   */
  getClient(): S3Client {
    return this.client;
  }

  /**
   * Get the bucket name
   */
  getBucket(): string {
    return this.bucket;
  }
}

// Export singleton instance
export const s3Service = new S3Service();
