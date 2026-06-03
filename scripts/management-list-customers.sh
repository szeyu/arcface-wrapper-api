#!/bin/bash
# List all enrolled customers with pagination

set -euo pipefail

LIMIT=${1:-50}
OFFSET=${2:-0}
API_URL=${3:-"http://localhost:3000"}

echo "Listing enrolled customers..."
echo "Limit: $LIMIT, Offset: $OFFSET"
echo "API: $API_URL/api/management/customers"
echo ""

curl -fsS "$API_URL/api/management/customers?limit=$LIMIT&offset=$OFFSET" | jq '.'

echo ""
echo "✓ Customer list retrieved"
