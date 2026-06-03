#!/bin/bash

# Script to test POST /faces/enroll endpoint
# Usage: ./scripts/faces-enroll.sh <face_id> <customer_identifier> [customer_name] [api_url]

set -euo pipefail

FACE_ID="${1:-}"
CUSTOMER_IDENTIFIER="${2:-}"
CUSTOMER_NAME="${3:-}"
API_URL="${4:-http://localhost:3000}"

if [ -z "$FACE_ID" ] || [ -z "$CUSTOMER_IDENTIFIER" ]; then
  echo "Usage: $0 <face_id> <customer_identifier> [customer_name] [api_url]"
  echo "Example: $0 abc123-def456 CUST001 \"John Doe\""
  exit 1
fi

echo "Enrolling customer..."
echo "Face ID: $FACE_ID"
echo "Customer Identifier: $CUSTOMER_IDENTIFIER"
if [ -n "$CUSTOMER_NAME" ]; then
  echo "Customer Name: $CUSTOMER_NAME"
fi
echo "API: $API_URL/api/faces/enroll"
echo ""

# Build JSON payload
if [ -n "$CUSTOMER_NAME" ]; then
  PAYLOAD=$(jq -n \
    --arg face_id "$FACE_ID" \
    --arg customer_identifier "$CUSTOMER_IDENTIFIER" \
    --arg customer_name "$CUSTOMER_NAME" \
    '{face_id: $face_id, customer_identifier: $customer_identifier, customer_name: $customer_name}')
else
  PAYLOAD=$(jq -n \
    --arg face_id "$FACE_ID" \
    --arg customer_identifier "$CUSTOMER_IDENTIFIER" \
    '{face_id: $face_id, customer_identifier: $customer_identifier}')
fi

# Send enrollment request
curl -fsS -X POST "$API_URL/api/faces/enroll" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD" \
  | jq .

echo ""
echo "✓ Customer enrolled successfully"
