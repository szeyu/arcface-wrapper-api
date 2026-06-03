#!/bin/bash
# Delete an enrolled customer by customer_id

set -euo pipefail

if [ -z "$1" ]; then
  echo "Usage: $0 <customer_id> [api_url]"
  echo "Example: $0 abc-123-def-456"
  echo ""
  echo "⚠️  WARNING: This will permanently delete:"
  echo "  - Customer enrollment record from database"
  echo "  - Customer embedding vector"
  echo ""
  echo "Note: This does NOT delete the detected face or S3 images"
  exit 1
fi

CUSTOMER_ID=$1
API_URL=${2:-"http://localhost:3000"}

echo "⚠️  Deleting customer..."
echo "Customer ID: $CUSTOMER_ID"
echo "API: $API_URL/api/management/customers/$CUSTOMER_ID"
echo ""
read -p "Are you sure you want to delete this customer? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
  echo "Deletion cancelled"
  exit 0
fi

echo ""
curl -fsS -X DELETE "$API_URL/api/management/customers/$CUSTOMER_ID" | jq '.'

echo ""
echo "✓ Customer deletion complete"
