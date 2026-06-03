/**
 * Example fixture coverage.
 *
 * This test intentionally touches every file in examples/ so new or renamed
 * examples cannot silently sit outside the verification suite.
 */

import request from 'supertest';
import { Express } from 'express';
import { promises as fs } from 'fs';
import path from 'path';

let app: Express;
const detectedFaceIds: string[] = [];

interface ExampleFixture {
  fileName: string;
  minFaces: number;
  expectedStatus: 200 | 400;
}

const fixtures: ExampleFixture[] = [
  { fileName: 'elon_musk_enroll.jpg', minFaces: 1, expectedStatus: 200 },
  { fileName: 'elon_musk_jensen_huang_mixed_large.avif', minFaces: 1, expectedStatus: 200 },
  { fileName: 'elon_musk_jensen_huang_mixed_large.webp', minFaces: 1, expectedStatus: 200 },
  { fileName: 'elon_musk_positive.jpg', minFaces: 1, expectedStatus: 200 },
  { fileName: 'elon_musk_profile.jpg', minFaces: 1, expectedStatus: 200 },
  { fileName: 'elon_musk_trump_mixed_profile.jpg', minFaces: 1, expectedStatus: 200 },
  { fileName: 'elon_musk_trump_mixed_small.jpg', minFaces: 2, expectedStatus: 200 },
  { fileName: 'jensen_huang_elon_musk_mixed_small.jpeg', minFaces: 1, expectedStatus: 200 },
  { fileName: 'jensen_huang_enroll.jpg', minFaces: 1, expectedStatus: 200 },
  { fileName: 'jensen_huang_positive.jpg', minFaces: 1, expectedStatus: 200 },
  { fileName: 'no_face_box.jpeg', minFaces: 0, expectedStatus: 400 },
  { fileName: 'trump_jensen_huang_mixed_profile.jpeg', minFaces: 1, expectedStatus: 200 },
  { fileName: 'trump_jensen_huang_mixed_small.jpeg', minFaces: 2, expectedStatus: 200 },
  { fileName: 'trump_xi_jinping_mixed.jpg', minFaces: 2, expectedStatus: 200 },
  { fileName: 'trump_xi_jinping_mixed_profile.webp', minFaces: 1, expectedStatus: 200 },
  { fileName: 'xi_jinping_solo.png', minFaces: 1, expectedStatus: 200 },
  { fileName: 'xi_jinping_trump_mixed_small.jpeg', minFaces: 2, expectedStatus: 200 },
];

beforeAll(async () => {
  const serverModule = await import('../server.js');
  const { client, connectDB } = await import('../db.js');
  const { initModels } = await import('../embedding.js');
  const { s3Service } = await import('../services/s3Service.js');

  app = serverModule.app;
  await connectDB();
  await client.query("DELETE FROM detected_faces WHERE identifier LIKE 'TEST_EXAMPLE_%'");
  await s3Service.ensureBucketExists();
  await initModels();
}, 60000);

afterAll(async () => {
  for (const faceId of detectedFaceIds) {
    try {
      await request(app).delete(`/api/management/faces/${faceId}`);
    } catch {
      // Keep cleanup best-effort so a failed delete does not mask the test failure.
    }
  }
}, 60000);

describe('Example fixture coverage', () => {
  it('should have every examples/ file represented in the fixture matrix', async () => {
    const examplesDir = path.join(process.cwd(), 'examples');
    const actualFiles = (await fs.readdir(examplesDir))
      .filter((fileName) => /\.(avif|jpe?g|png|webp)$/i.test(fileName))
      .sort();

    const expectedFiles = fixtures.map((fixture) => fixture.fileName).sort();

    expect(actualFiles).toEqual(expectedFiles);
  });

  it.each(fixtures)('should run detect coverage for $fileName', async ({ fileName, minFaces, expectedStatus }) => {
    const imagePath = path.join(process.cwd(), 'examples', fileName);
    const imageBuffer = await fs.readFile(imagePath);

    const response = await request(app)
      .post('/api/faces/detect')
      .attach('file', imageBuffer, fileName)
      .field('identifier', `TEST_EXAMPLE_${fileName}`)
      .expect(expectedStatus);

    if (expectedStatus === 400) {
      expect(response.body).toHaveProperty('error');
      return;
    }

    expect(response.body).toBeInstanceOf(Array);
    expect(response.body.length).toBeGreaterThanOrEqual(minFaces);

    for (const face of response.body) {
      detectedFaceIds.push(face.face_id);
    }
  }, 45000);
});
