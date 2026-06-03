#!/usr/bin/env node

import { mkdir, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { basename, join } from "node:path";
import { performance } from "node:perf_hooks";

const apiUrl = process.argv[2] ?? "http://localhost:3000";
const iterations = Number.parseInt(process.argv[3] ?? "3", 10);
const outputDir = process.argv[4] ?? "benchmarks/results";

const runId = `BENCH_${new Date().toISOString().replace(/[:.]/g, "-")}`;
const enrolledCustomerIds = [];
const detectedFaceIds = [];

const examples = {
  elonEnroll: "examples/elon_musk_enroll.jpg",
  elonPositive: "examples/elon_musk_positive.jpg",
  jensenEnroll: "examples/jensen_huang_enroll.jpg",
  jensenPositive: "examples/jensen_huang_positive.jpg",
  negative: "examples/xi_jinping_solo.png",
  multiFaceJpeg: "examples/elon_musk_trump_mixed_small.jpg",
  webp: "examples/elon_musk_jensen_huang_mixed_large.webp",
  avif: "examples/elon_musk_jensen_huang_mixed_large.avif",
};

const assertExamplesExist = () => {
  for (const path of Object.values(examples)) {
    if (!existsSync(path)) {
      throw new Error(`Missing benchmark image: ${path}`);
    }
  }
};

const round = (value) => Math.round(value * 100) / 100;

const timed = async (label, fn) => {
  const startedAt = performance.now();
  const result = await fn();
  return {
    label,
    ms: round(performance.now() - startedAt),
    result,
  };
};

const requestJson = async (path, options = {}) => {
  const response = await fetch(`${apiUrl}${path}`, options);
  const text = await response.text();
  const body = text ? JSON.parse(text) : null;

  if (!response.ok) {
    throw new Error(`${options.method ?? "GET"} ${path} failed with ${response.status}: ${text}`);
  }

  return body;
};

const uploadImage = async (imagePath, identifier) => {
  const { readFile } = await import("node:fs/promises");
  const buffer = await readFile(imagePath);
  const uploadForm = new FormData();
  uploadForm.append("file", new Blob([buffer]), basename(imagePath));
  uploadForm.append("identifier", identifier);

  const faces = await requestJson("/api/faces/detect", {
    method: "POST",
    body: uploadForm,
  });

  for (const face of faces) {
    if (face.face_id) {
      detectedFaceIds.push(face.face_id);
    }
  }

  return faces;
};

const enrollCustomer = async (faceId, customerIdentifier, customerName) => {
  const body = await requestJson("/api/faces/enroll", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      face_id: faceId,
      customer_identifier: customerIdentifier,
      customer_name: customerName,
    }),
  });

  enrolledCustomerIds.push(body.customer_id);
  return body;
};

const recognizeFace = (faceId, minConfidence = 0.5) =>
  requestJson("/api/faces/recognize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ face_id: faceId, min_confidence: minConfidence }),
  });

const deleteCustomer = async (customerId) => {
  await requestJson(`/api/management/customers/${customerId}`, { method: "DELETE" });
};

const deleteFace = async (faceId) => {
  await requestJson(`/api/management/faces/${faceId}`, { method: "DELETE" });
};

const cleanup = async () => {
  for (const customerId of enrolledCustomerIds) {
    try {
      await deleteCustomer(customerId);
    } catch {
      // Continue cleanup even if a customer was already deleted.
    }
  }

  try {
    const customers = await requestJson("/api/management/customers?limit=1000");
    for (const customer of customers.customers ?? []) {
      if (customer.customer_identifier?.startsWith("BENCH_")) {
        await deleteCustomer(customer.customer_id);
      }
    }
  } catch {
    // If the API is down, report the benchmark error instead of masking it.
  }

  for (const faceId of detectedFaceIds) {
    try {
      await deleteFace(faceId);
    } catch {
      // Continue cleanup even if a face was already deleted or not deletable.
    }
  }
};

const percentile = (values, p) => {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, Math.min(index, sorted.length - 1))];
};

const summarize = (samples) => {
  const values = samples.map((sample) => sample.ms);
  const total = values.reduce((sum, value) => sum + value, 0);
  return {
    count: values.length,
    min_ms: round(Math.min(...values)),
    p50_ms: round(percentile(values, 50)),
    p95_ms: round(percentile(values, 95)),
    max_ms: round(Math.max(...values)),
    avg_ms: round(total / values.length),
  };
};

const markdownReport = (report) => {
  const rows = Object.entries(report.summary)
    .map(
      ([name, stats]) =>
        `| ${name} | ${stats.count} | ${stats.min_ms} | ${stats.p50_ms} | ${stats.p95_ms} | ${stats.max_ms} | ${stats.avg_ms} |`
    )
    .join("\n");

  const matches = report.recognition_results
    .map(
      (item) =>
        `| ${item.scenario} | ${item.expected} | ${item.top_match ?? "no match"} | ${item.confidence_score ?? ""} |`
    )
    .join("\n");

  return `# FaceVector Benchmark Report

- Timestamp: ${report.timestamp}
- API URL: ${report.api_url}
- Iterations: ${report.iterations}
- Node: ${report.environment.node}
- Platform: ${report.environment.platform}

## Timing Summary

| Operation | Count | Min ms | P50 ms | P95 ms | Max ms | Avg ms |
|-----------|-------|--------|--------|--------|--------|--------|
${rows}

## Recognition Results

| Scenario | Expected | Top match | Confidence |
|----------|----------|-----------|------------|
${matches}
`;
};

const main = async () => {
  if (!Number.isInteger(iterations) || iterations < 1) {
    throw new Error("Iterations must be a positive integer");
  }

  assertExamplesExist();
  await mkdir(outputDir, { recursive: true });

  const samples = {
    detect_single_face_jpeg: [],
    detect_multi_face_jpeg: [],
    detect_webp: [],
    detect_avif: [],
    recognize: [],
    full_positive_flow: [],
    full_negative_flow: [],
  };
  const recognitionResults = [];

  const elonEnrollDetection = await timed("setup_detect_elon", () =>
    uploadImage(examples.elonEnroll, `${runId}_ELON_ENROLL`)
  );
  const jensenEnrollDetection = await timed("setup_detect_jensen", () =>
    uploadImage(examples.jensenEnroll, `${runId}_JENSEN_ENROLL`)
  );

  const elonFaceId = elonEnrollDetection.result[0]?.face_id;
  const jensenFaceId = jensenEnrollDetection.result[0]?.face_id;

  if (!elonFaceId || !jensenFaceId) {
    throw new Error("Setup failed: expected enrollment faces were not detected");
  }

  await timed("setup_enroll_elon", () => enrollCustomer(elonFaceId, `${runId}_ELON`, "Benchmark Elon"));
  await timed("setup_enroll_jensen", () => enrollCustomer(jensenFaceId, `${runId}_JENSEN`, "Benchmark Jensen"));

  for (let i = 0; i < iterations; i += 1) {
    samples.detect_single_face_jpeg.push(
      await timed("detect_single_face_jpeg", () => uploadImage(examples.elonPositive, `${runId}_JPEG_SINGLE_${i}`))
    );
    samples.detect_multi_face_jpeg.push(
      await timed("detect_multi_face_jpeg", () => uploadImage(examples.multiFaceJpeg, `${runId}_JPEG_MULTI_${i}`))
    );
    samples.detect_webp.push(
      await timed("detect_webp", () => uploadImage(examples.webp, `${runId}_WEBP_${i}`))
    );
    samples.detect_avif.push(
      await timed("detect_avif", () => uploadImage(examples.avif, `${runId}_AVIF_${i}`))
    );

    const positiveFlow = await timed("full_positive_flow", async () => {
      const faces = await uploadImage(examples.jensenPositive, `${runId}_JENSEN_POS_${i}`);
      const faceId = faces[0]?.face_id;
      const matches = await recognizeFace(faceId);
      return matches;
    });
    samples.full_positive_flow.push(positiveFlow);

    const topPositive = positiveFlow.result[0];
    recognitionResults.push({
      scenario: `jensen_positive_${i + 1}`,
      expected: `${runId}_JENSEN`,
      top_match: topPositive?.customer_identifier,
      confidence_score: topPositive?.confidence_score,
    });

    const recognizeOnly = await timed("recognize", () => recognizeFace(elonFaceId));
    samples.recognize.push(recognizeOnly);

    const negativeFlow = await timed("full_negative_flow", async () => {
      const faces = await uploadImage(examples.negative, `${runId}_NEG_${i}`);
      const faceId = faces[0]?.face_id;
      return recognizeFace(faceId);
    });
    samples.full_negative_flow.push(negativeFlow);

    recognitionResults.push({
      scenario: `xi_negative_${i + 1}`,
      expected: "no match",
      top_match: negativeFlow.result[0]?.customer_identifier,
      confidence_score: negativeFlow.result[0]?.confidence_score,
    });
  }

  const summary = Object.fromEntries(
    Object.entries(samples).map(([name, operationSamples]) => [name, summarize(operationSamples)])
  );

  const report = {
    timestamp: new Date().toISOString(),
    api_url: apiUrl,
    iterations,
    run_id: runId,
    environment: {
      node: process.version,
      platform: `${process.platform} ${process.arch}`,
    },
    summary,
    recognition_results: recognitionResults,
  };

  const reportBase = join(outputDir, `${runId.toLowerCase()}`);
  await writeFile(`${reportBase}.json`, JSON.stringify(report, null, 2));
  await writeFile(`${reportBase}.md`, markdownReport(report));

  console.log(JSON.stringify(report.summary, null, 2));
  console.log(`Benchmark report written to ${reportBase}.json and ${reportBase}.md`);
};

try {
  await main();
} finally {
  await cleanup();
}
