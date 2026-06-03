#!/bin/bash

# Script to test GET /faces/{face_id} endpoint
# Usage: ./scripts/faces-get-image.sh <face_id> [output_path] [api_url]

set -euo pipefail

FACE_ID="${1:-}"
OUTPUT_PATH="${2:-output/retrieved_face.jpg}"
API_URL="${3:-http://localhost:3000}"

if [ -z "$FACE_ID" ]; then
  echo "Usage: $0 <face_id> [output_path] [api_url]"
  echo "Example: $0 abc123-def456-ghi789 output/face.jpg"
  exit 1
fi

echo "Retrieving face image: $FACE_ID"
echo "API: $API_URL/api/faces/$FACE_ID"
echo "Output: $OUTPUT_PATH"
echo ""

# Create output directory if needed
mkdir -p "$(dirname "$OUTPUT_PATH")"

# Download face image
curl -fsS -X GET "$API_URL/api/faces/$FACE_ID" \
  -H "Accept: image/jpeg" \
  -o "$OUTPUT_PATH"

echo ""
echo "✓ Face image saved to: $OUTPUT_PATH"
