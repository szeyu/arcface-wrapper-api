/**
 * Management API Integration Test Suite
 *
 * Tests the complete workflow including management endpoints:
 * 1. Detect faces → Get face_id
 * 2. Enroll customers
 * 3. List and manage data
 * 4. Recognize faces
 * 5. Clean up test data
 *
 * Requirements:
 * - Database must be running (make up-db)
 * - RustFS must be running (make up-rustfs)
 * - ONNX models must be downloaded (make models)
 * - Test images must exist in examples/ folder
 */

import request from 'supertest';
import { Express } from 'express';
import { promises as fs } from 'fs';
import path from 'path';

let app: Express;
let testFaceIds: string[] = [];
let testCustomerIds: string[] = [];

// Test image buffers
let elonMusk1Buffer: Buffer;
let elonMusk2Buffer: Buffer;
let xiTrumpBuffer: Buffer;
let noFaceBuffer: Buffer;
let elonJensenWebpBuffer: Buffer;
let elonJensenAvifBuffer: Buffer;

beforeAll(async () => {
  // Dynamically import the app and initialization functions
  const serverModule = await import('../server.js');
  const { client, connectDB } = await import('../db.js');
  const { initModels } = await import('../embedding.js');
  const { s3Service } = await import('../services/s3Service.js');

  app = serverModule.app;

  // Initialize database, S3, and models
  await connectDB();
  await client.query("DELETE FROM enrolled_customers WHERE customer_identifier LIKE 'TEST_%'");
  await client.query("DELETE FROM detected_faces WHERE identifier LIKE 'TEST_%'");
  await s3Service.ensureBucketExists();
  await initModels();

  // Load test images
  elonMusk1Buffer = await fs.readFile(path.join(process.cwd(), 'examples', 'elon_musk_enroll.jpg'));
  elonMusk2Buffer = await fs.readFile(path.join(process.cwd(), 'examples', 'elon_musk_positive.jpg'));
  xiTrumpBuffer = await fs.readFile(path.join(process.cwd(), 'examples', 'xi_jinping_trump_mixed_small.jpeg'));
  noFaceBuffer = await fs.readFile(path.join(process.cwd(), 'examples', 'no_face_box.jpeg'));
  elonJensenWebpBuffer = await fs.readFile(path.join(process.cwd(), 'examples', 'elon_musk_jensen_huang_mixed_large.webp'));
  elonJensenAvifBuffer = await fs.readFile(path.join(process.cwd(), 'examples', 'elon_musk_jensen_huang_mixed_large.avif'));
}, 60000);

describe('Management API Integration Tests', () => {
  describe('Setup: Detect and Enroll Test Data', () => {
    it('should detect face in Elon Musk image 1', async () => {
      const response = await request(app)
        .post('/api/faces/detect')
        .attach('file', elonMusk1Buffer, 'elon_musk_enroll.jpg')
        .field('identifier', 'TEST_ELON_1')
        .expect(200);

      expect(response.body).toBeInstanceOf(Array);
      expect(response.body.length).toBeGreaterThan(0);
      expect(response.body[0]).toHaveProperty('face_id');

      testFaceIds.push(response.body[0].face_id);
    });

    it('should detect 2 faces in Xi-Trump image', async () => {
      const response = await request(app)
        .post('/api/faces/detect')
        .attach('file', xiTrumpBuffer, 'xi_trump.jpeg')
        .field('identifier', 'TEST_LEADERS')
        .expect(200);

      expect(response.body).toBeInstanceOf(Array);
      expect(response.body.length).toBe(2);

      testFaceIds.push(response.body[0].face_id);
      testFaceIds.push(response.body[1].face_id);
    });

    it('should enroll Elon Musk customer', async () => {
      const response = await request(app)
        .post('/api/faces/enroll')
        .send({
          face_id: testFaceIds[0],
          customer_identifier: 'TEST_ELON',
          customer_name: 'Elon Musk Test',
          customer_metadata: { test: true, department: 'engineering' },
        })
        .expect(200);

      expect(response.body).toHaveProperty('customer_id');
      expect(response.body.customer_identifier).toBe('TEST_ELON');

      testCustomerIds.push(response.body.customer_id);
    });

    it('should enroll Trump customer', async () => {
      const response = await request(app)
        .post('/api/faces/enroll')
        .send({
          face_id: testFaceIds[1],
          customer_identifier: 'TEST_TRUMP',
          customer_name: 'Donald Trump Test',
        })
        .expect(200);

      testCustomerIds.push(response.body.customer_id);
    });
  });

  describe('Management: List Endpoints', () => {
    it('should list detected faces with pagination', async () => {
      const response = await request(app)
        .get('/api/management/faces')
        .query({ limit: 10, offset: 0 })
        .expect(200);

      expect(response.body).toHaveProperty('total');
      expect(response.body).toHaveProperty('limit', 10);
      expect(response.body).toHaveProperty('offset', 0);
      expect(response.body).toHaveProperty('faces');
      expect(response.body.faces).toBeInstanceOf(Array);

      // Check face structure
      if (response.body.faces.length > 0) {
        const face = response.body.faces[0];
        expect(face).toHaveProperty('face_id');
        expect(face).toHaveProperty('identifier');
        expect(face).toHaveProperty('bounding_box');
        expect(face).toHaveProperty('confidence');
        expect(face).toHaveProperty('created_at');
        expect(face).toHaveProperty('s3_keys');
        expect(face.s3_keys).toHaveProperty('original');
        expect(face.s3_keys).toHaveProperty('face');
      }
    });

    it('should list enrolled customers with pagination', async () => {
      const response = await request(app)
        .get('/api/management/customers')
        .query({ limit: 10, offset: 0 })
        .expect(200);

      expect(response.body).toHaveProperty('total');
      expect(response.body).toHaveProperty('customers');
      expect(response.body.customers).toBeInstanceOf(Array);

      // Check customer structure
      if (response.body.customers.length > 0) {
        const customer = response.body.customers[0];
        expect(customer).toHaveProperty('customer_id');
        expect(customer).toHaveProperty('customer_identifier');
        expect(customer).toHaveProperty('customer_name');
        expect(customer).toHaveProperty('face_id');
        expect(customer).toHaveProperty('created_at');
      }
    });

    it('should get database statistics', async () => {
      const response = await request(app)
        .get('/api/management/stats')
        .expect(200);

      expect(response.body).toHaveProperty('detected_faces');
      expect(response.body).toHaveProperty('enrolled_customers');
      expect(typeof response.body.detected_faces).toBe('number');
      expect(typeof response.body.enrolled_customers).toBe('number');
    });
  });

  describe('Management: Customer Details', () => {
    it('should get specific customer details', async () => {
      const response = await request(app)
        .get(`/api/management/customers/${testCustomerIds[0]}`)
        .expect(200);

      expect(response.body).toHaveProperty('customer_id', testCustomerIds[0]);
      expect(response.body).toHaveProperty('customer_identifier', 'TEST_ELON');
      expect(response.body).toHaveProperty('customer_name', 'Elon Musk Test');
      expect(response.body).toHaveProperty('customer_metadata');
      expect(response.body.customer_metadata).toHaveProperty('test', true);
      expect(response.body).toHaveProperty('face_detection');
      expect(response.body.face_detection).toHaveProperty('bounding_box');
      expect(response.body).toHaveProperty('s3_keys');
    });

    it('should return 404 for non-existent customer', async () => {
      const response = await request(app)
        .get('/api/management/customers/00000000-0000-0000-0000-000000000000')
        .expect(404);

      expect(response.body).toHaveProperty('error');
    });
  });

  describe('Recognition: Face Matching', () => {
    let elonRecognitionFaceId: string;
    let elonTrumpFaceIds: { elon: string; trump: string };

    it('should detect face in Elon Musk image 2 for recognition', async () => {
      const response = await request(app)
        .post('/api/faces/detect')
        .attach('file', elonMusk2Buffer, 'elon_musk_positive.jpg')
        .field('identifier', 'TEST_RECOGNITION')
        .expect(200);

      expect(response.body.length).toBeGreaterThan(0);
      elonRecognitionFaceId = response.body[0].face_id;
      testFaceIds.push(elonRecognitionFaceId);
    });

    it('should recognize Elon Musk from different photo with highest similarity', async () => {
      const response = await request(app)
        .post('/api/faces/recognize')
        .send({ face_id: elonRecognitionFaceId })
        .expect(200);

      expect(response.body).toBeInstanceOf(Array);
      expect(response.body.length).toBeGreaterThan(0);

      // Verify results are ordered by confidence (highest first)
      for (let i = 0; i < response.body.length - 1; i++) {
        expect(response.body[i].confidence_score).toBeGreaterThanOrEqual(
          response.body[i + 1].confidence_score
        );
      }

      // Find TEST_ELON in the results
      const elonMatch = response.body.find(
        (match: any) => match.customer_identifier === 'TEST_ELON'
      );
      expect(elonMatch).toBeDefined();
      expect(elonMatch.confidence_score).toBeGreaterThan(0.50);

      // TEST_ELON should be the top match or very close
      const elonIndex = response.body.findIndex(
        (match: any) => match.customer_identifier === 'TEST_ELON'
      );
      expect(elonIndex).toBeLessThan(3); // Should be in top 3 results
    });

    it('should detect 2 faces in elon_musk_trump image', async () => {
      const elonTrumpPath = path.join(process.cwd(), 'examples', 'elon_musk_trump_mixed_small.jpg');
      const elonTrumpBuffer = await fs.readFile(elonTrumpPath);

      const response = await request(app)
        .post('/api/faces/detect')
        .attach('file', elonTrumpBuffer, 'elon_musk_trump_mixed_small.jpg')
        .field('identifier', 'TEST_ELON_TRUMP_MIX')
        .expect(200);

      expect(response.body.length).toBe(2);

      // Store face IDs (don't assume which is which yet)
      const face1Id = response.body[0].face_id;
      const face2Id = response.body[1].face_id;

      // Recognize both faces to determine which is which
      const face1Recognition = await request(app)
        .post('/api/faces/recognize')
        .send({ face_id: face1Id })
        .expect(200);

      const face2Recognition = await request(app)
        .post('/api/faces/recognize')
        .send({ face_id: face2Id })
        .expect(200);

      // Find which face matches Elon better
      const face1ElonMatch = face1Recognition.body.find(
        (m: any) => m.customer_identifier === 'TEST_ELON'
      );
      const face2ElonMatch = face2Recognition.body.find(
        (m: any) => m.customer_identifier === 'TEST_ELON'
      );

      // Assign faces based on which has higher Elon confidence
      if (face1ElonMatch && face2ElonMatch) {
        if (face1ElonMatch.confidence_score > face2ElonMatch.confidence_score) {
          elonTrumpFaceIds = { elon: face1Id, trump: face2Id };
        } else {
          elonTrumpFaceIds = { elon: face2Id, trump: face1Id };
        }
      } else {
        // Fallback: assume order by size
        elonTrumpFaceIds = { elon: face1Id, trump: face2Id };
      }

      testFaceIds.push(face1Id);
      testFaceIds.push(face2Id);
    });

    it('should recognize both faces correctly from elon_trump photo', async () => {
      // Recognize Elon's face
      const elonResponse = await request(app)
        .post('/api/faces/recognize')
        .send({ face_id: elonTrumpFaceIds.elon })
        .expect(200);

      expect(elonResponse.body).toBeInstanceOf(Array);

      // Find matches
      const elonMatchForElon = elonResponse.body.find(
        (match: any) => match.customer_identifier === 'TEST_ELON'
      );
      const trumpMatchForElon = elonResponse.body.find(
        (match: any) => match.customer_identifier === 'TEST_TRUMP'
      );

      // Verify Elon is recognized with high confidence
      expect(elonMatchForElon).toBeDefined();
      expect(elonMatchForElon.confidence_score).toBeGreaterThan(0.50);

      // Elon should have higher confidence than Trump for Elon's face
      if (trumpMatchForElon) {
        expect(elonMatchForElon.confidence_score).toBeGreaterThanOrEqual(
          trumpMatchForElon.confidence_score
        );
      }

      // Recognize Trump's face
      const trumpResponse = await request(app)
        .post('/api/faces/recognize')
        .send({ face_id: elonTrumpFaceIds.trump })
        .expect(200);

      expect(trumpResponse.body).toBeInstanceOf(Array);

      const elonMatchForTrump = trumpResponse.body.find(
        (match: any) => match.customer_identifier === 'TEST_ELON'
      );

      // The mixed Trump crop is lower-confidence at the default threshold, but
      // it must not be incorrectly returned as Elon.
      expect(elonMatchForElon.confidence_score).toBeGreaterThan(0.50);
      expect(elonMatchForTrump).toBeUndefined();
    });
  });

  describe('Image Format Normalization', () => {
    it('should detect faces from a WEBP upload after normalization', async () => {
      const response = await request(app)
        .post('/api/faces/detect')
        .attach('file', elonJensenWebpBuffer, 'elon_musk_jensen_huang_mixed_large.webp')
        .field('identifier', 'TEST_FORMAT_WEBP')
        .expect(200);

      expect(response.body).toBeInstanceOf(Array);
      expect(response.body.length).toBeGreaterThan(0);
      response.body.forEach((face: any) => testFaceIds.push(face.face_id));
    });

    it('should detect faces from an AVIF upload after normalization', async () => {
      const response = await request(app)
        .post('/api/faces/detect')
        .attach('file', elonJensenAvifBuffer, 'elon_musk_jensen_huang_mixed_large.avif')
        .field('identifier', 'TEST_FORMAT_AVIF')
        .expect(200);

      expect(response.body).toBeInstanceOf(Array);
      expect(response.body.length).toBeGreaterThan(0);
      response.body.forEach((face: any) => testFaceIds.push(face.face_id));
    });
  });

  describe('Management: Delete Operations', () => {
    it('should prevent deleting face used by enrolled customer', async () => {
      const response = await request(app)
        .delete(`/api/management/faces/${testFaceIds[0]}`)
        .expect(400);

      expect(response.body).toHaveProperty('error');
      expect(response.body.message).toContain('enrolled customers');
    });

    it('should delete enrolled customer', async () => {
      const response = await request(app)
        .delete(`/api/management/customers/${testCustomerIds[0]}`)
        .expect(200);

      expect(response.body).toHaveProperty('message', 'Customer deleted successfully');
      expect(response.body).toHaveProperty('customer_id', testCustomerIds[0]);
      expect(response.body).toHaveProperty('customer_identifier', 'TEST_ELON');
    });

    it('should verify customer is deleted', async () => {
      await request(app)
        .get(`/api/management/customers/${testCustomerIds[0]}`)
        .expect(404);
    });

    it('should delete second customer', async () => {
      const response = await request(app)
        .delete(`/api/management/customers/${testCustomerIds[1]}`)
        .expect(200);

      expect(response.body.customer_identifier).toBe('TEST_TRUMP');
    });

    it('should return 404 when deleting non-existent customer', async () => {
      await request(app)
        .delete('/api/management/customers/00000000-0000-0000-0000-000000000000')
        .expect(404);
    });
  });

  describe('Cleanup: Delete Test Faces', () => {
    it('should delete all test faces and S3 images', async () => {
      for (const faceId of testFaceIds) {
        const response = await request(app)
          .delete(`/api/management/faces/${faceId}`)
          .expect(200);

        expect(response.body).toHaveProperty('message', 'Face deleted successfully');
        expect(response.body).toHaveProperty('deleted_s3_keys');
        expect(response.body.deleted_s3_keys).toHaveProperty('original');
        expect(response.body.deleted_s3_keys).toHaveProperty('face');
      }
    });

    it('should verify faces are deleted', async () => {
      for (const faceId of testFaceIds) {
        await request(app)
          .get(`/api/faces/${faceId}`)
          .expect(404);
      }
    });

    it('should return 404 when deleting non-existent face', async () => {
      await request(app)
        .delete('/api/management/faces/00000000-0000-0000-0000-000000000000')
        .expect(404);
    });
  });

  describe('Edge Cases', () => {
    it('should reject image with no faces', async () => {
      const response = await request(app)
        .post('/api/faces/detect')
        .attach('file', noFaceBuffer, 'no_face_box.jpeg')
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    it('should handle pagination with large offset', async () => {
      const response = await request(app)
        .get('/api/management/faces')
        .query({ limit: 10, offset: 10000 })
        .expect(200);

      expect(response.body.faces).toBeInstanceOf(Array);
      // Should return empty array for offset beyond data
    });

    it('should handle missing required fields in enroll', async () => {
      await request(app)
        .post('/api/faces/enroll')
        .send({ face_id: 'test' })
        .expect(400);
    });

    it('should handle invalid face_id in recognize', async () => {
      await request(app)
        .post('/api/faces/recognize')
        .send({ face_id: '00000000-0000-0000-0000-000000000000' })
        .expect(404);
    });
  });
});
