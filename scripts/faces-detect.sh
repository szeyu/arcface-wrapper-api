#!/bin/bash

# Script to test POST /faces/detect endpoint
# Usage: ./scripts/faces-detect.sh <image_path> [identifier] [api_url]

set -euo pipefail

IMAGE_PATH="${1:-}"
IDENTIFIER="${2:-}"
API_URL="${3:-http://localhost:3000}"

if [ -z "$IMAGE_PATH" ]; then
  echo "Usage: $0 <image_path> [identifier] [api_url]"
  echo "Example: $0 examples/elon_musk_enroll.jpg customer_123"
  exit 1
fi

if [ ! -f "$IMAGE_PATH" ]; then
  echo "Error: Image file not found: $IMAGE_PATH"
  exit 1
fi

echo "Detecting faces in: $IMAGE_PATH"
if [ -n "$IDENTIFIER" ]; then
  echo "Identifier: $IDENTIFIER"
fi
echo "API: $API_URL/api/faces/detect"
echo ""

# Send multipart form-data request
if [ -n "$IDENTIFIER" ]; then
  curl -fsS -X POST "$API_URL/api/faces/detect" \
    -F "file=@$IMAGE_PATH" \
    -F "identifier=$IDENTIFIER" \
    -H "Accept: application/json" \
    | jq .
else
  curl -fsS -X POST "$API_URL/api/faces/detect" \
    -F "file=@$IMAGE_PATH" \
    -H "Accept: application/json" \
    | jq .
fi

echo ""
echo "✓ Face detection complete"
