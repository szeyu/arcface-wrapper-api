#!/bin/bash

# Script to test POST /faces/recognize endpoint
# Usage: ./scripts/faces-recognize.sh <face_id> [api_url] [min_confidence]

set -euo pipefail

FACE_ID="${1:-}"
API_URL="${2:-http://localhost:3000}"
MIN_CONFIDENCE="${3:-}"

if [ -z "$FACE_ID" ]; then
  echo "Usage: $0 <face_id> [api_url] [min_confidence]"
  echo "Example: $0 abc123-def456-ghi789 http://localhost:3000 0.5"
  exit 1
fi

echo "Recognizing face: $FACE_ID"
echo "API: $API_URL/api/faces/recognize"
if [ -n "$MIN_CONFIDENCE" ]; then
  echo "Minimum confidence: $MIN_CONFIDENCE"
fi
echo ""

# Build JSON payload
if [ -n "$MIN_CONFIDENCE" ]; then
  PAYLOAD=$(jq -n \
    --arg face_id "$FACE_ID" \
    --argjson min_confidence "$MIN_CONFIDENCE" \
    '{face_id: $face_id, min_confidence: $min_confidence}')
else
  PAYLOAD=$(jq -n \
    --arg face_id "$FACE_ID" \
    '{face_id: $face_id}')
fi

# Send recognition request
curl -fsS -X POST "$API_URL/api/faces/recognize" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD" \
  | jq .

echo ""
echo "✓ Recognition complete"
