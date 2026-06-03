#!/bin/bash
# List all detected faces with pagination

set -euo pipefail

LIMIT=${1:-50}
OFFSET=${2:-0}
API_URL=${3:-"http://localhost:3000"}

echo "Listing detected faces..."
echo "Limit: $LIMIT, Offset: $OFFSET"
echo "API: $API_URL/api/management/faces"
echo ""

curl -fsS "$API_URL/api/management/faces?limit=$LIMIT&offset=$OFFSET" | jq '.'

echo ""
echo "✓ Face list retrieved"
