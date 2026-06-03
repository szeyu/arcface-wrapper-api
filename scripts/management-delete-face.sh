#!/bin/bash
# Delete a detected face by face_id (also deletes S3 images)

set -euo pipefail

if [ -z "$1" ]; then
  echo "Usage: $0 <face_id> [api_url]"
  echo "Example: $0 abc-123-def-456"
  echo ""
  echo "⚠️  WARNING: This will permanently delete:"
  echo "  - Face metadata from database"
  echo "  - Original image from S3"
  echo "  - Cropped face image from S3"
  echo ""
  echo "Note: Cannot delete if face is used by enrolled customers"
  exit 1
fi

FACE_ID=$1
API_URL=${2:-"http://localhost:3000"}

echo "⚠️  Deleting face..."
echo "Face ID: $FACE_ID"
echo "API: $API_URL/api/management/faces/$FACE_ID"
echo ""
read -p "Are you sure you want to delete this face? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
  echo "Deletion cancelled"
  exit 0
fi

echo ""
curl -fsS -X DELETE "$API_URL/api/management/faces/$FACE_ID" | jq '.'

echo ""
echo "✓ Face deletion complete"
