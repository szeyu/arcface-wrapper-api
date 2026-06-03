#!/bin/bash
# Get customer details by customer_id

set -euo pipefail

if [ -z "$1" ]; then
  echo "Usage: $0 <customer_id> [api_url]"
  echo "Example: $0 abc-123-def-456"
  exit 1
fi

CUSTOMER_ID=$1
API_URL=${2:-"http://localhost:3000"}

echo "Getting customer details..."
echo "Customer ID: $CUSTOMER_ID"
echo "API: $API_URL/api/management/customers/$CUSTOMER_ID"
echo ""

curl -fsS "$API_URL/api/management/customers/$CUSTOMER_ID" | jq '.'

echo ""
echo "✓ Customer details retrieved"
