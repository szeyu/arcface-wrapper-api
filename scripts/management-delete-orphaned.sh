#!/bin/bash
# Delete all orphaned faces (faces not enrolled with any customer)

set -euo pipefail

API_URL=${1:-"http://localhost:3000"}

echo "⚠️  Deleting all orphaned faces..."
echo "API: $API_URL/api/management/faces/orphaned"
echo ""
echo "This will permanently delete:"
echo "  - All face detection records not linked to enrolled customers"
echo "  - Original images from S3"
echo "  - Cropped face images from S3"
echo ""
read -p "Are you sure you want to delete all orphaned faces? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
  echo "Deletion cancelled"
  exit 0
fi

echo ""
curl -fsS -X DELETE "$API_URL/api/management/faces/orphaned" | jq '.'

echo ""
echo "✓ Orphaned faces deletion complete"
