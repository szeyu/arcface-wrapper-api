import { randomUUID } from "crypto";
import { describe, expect, it } from "vitest";
import { s3Service } from "../services/s3Service.js";

describe("RustFS S3-compatible storage", () => {
  it("creates bucket, uploads, downloads, and deletes an object", async () => {
    const key = `test-storage/${randomUUID()}.txt`;
    const payload = Buffer.from(`facevector-storage-test-${randomUUID()}`, "utf8");

    await s3Service.ensureBucketExists();

    const uploadedKey = await s3Service.uploadImage(key, payload, "text/plain");
    expect(uploadedKey).toBe(key);

    const downloaded = await s3Service.downloadImage(key);
    expect(downloaded.toString("utf8")).toBe(payload.toString("utf8"));

    await s3Service.deleteImage(key);
    await expect(s3Service.downloadImage(key)).rejects.toThrow("S3 download failed");
  });
});
