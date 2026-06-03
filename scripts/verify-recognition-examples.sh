#!/bin/bash

# End-to-end recognition smoke test using curated examples.
# Usage: ./scripts/verify-recognition-examples.sh [api_url] [min_confidence]

set -euo pipefail

API_URL="${1:-http://localhost:3000}"
MIN_CONFIDENCE="${2:-0.5}"
RUN_ID="example_$(date +%s)"
ENROLLED_CUSTOMER_IDS=()

ELON_ENROLL_IMAGE="examples/elon_musk_enroll.jpg"
ELON_POSITIVE_IMAGE="examples/elon_musk_positive.jpg"
JENSEN_ENROLL_IMAGE="examples/jensen_huang_enroll.jpg"
JENSEN_POSITIVE_IMAGE="examples/jensen_huang_positive.jpg"
NEGATIVE_IMAGE="examples/xi_jinping_solo.png"

detect_face() {
  local image_path="$1"
  local identifier="$2"
  local index="${3:-0}"

  curl -sS -X POST "$API_URL/api/faces/detect" \
    -F "file=@$image_path" \
    -F "identifier=$identifier" \
    -H "Accept: application/json" \
    | jq -r ".[$index].face_id"
}

recognize_face() {
  local face_id="$1"

  jq -n \
    --arg face_id "$face_id" \
    --argjson min_confidence "$MIN_CONFIDENCE" \
    '{face_id: $face_id, min_confidence: $min_confidence}' \
    | curl -sS -X POST "$API_URL/api/faces/recognize" \
      -H "Content-Type: application/json" \
      -d @-
}

cleanup_examples() {
  local customer_ids

  customer_ids="$(curl -sS "$API_URL/api/management/customers?limit=1000" \
    | jq -r '.customers[] | select(.customer_identifier | startswith("example_") or startswith("MATRIX_")) | .customer_id')"

  if [ -n "$customer_ids" ]; then
    while IFS= read -r customer_id; do
      [ -z "$customer_id" ] && continue
      curl -sS -X DELETE "$API_URL/api/management/customers/$customer_id" >/dev/null || true
    done <<< "$customer_ids"
  fi

  curl -sS -X DELETE "$API_URL/api/management/faces/orphaned" >/dev/null || true
}

cleanup_current_examples() {
  for customer_id in "${ENROLLED_CUSTOMER_IDS[@]}"; do
    [ -z "$customer_id" ] && continue
    curl -sS -X DELETE "$API_URL/api/management/customers/$customer_id" >/dev/null || true
  done

  cleanup_examples
}

trap cleanup_current_examples EXIT

enroll_customer() {
  local image_path="$1"
  local suffix="$2"
  local customer_name="$3"
  local face_id
  local response
  local customer_id

  face_id="$(detect_face "$image_path" "${RUN_ID}_${suffix}_enroll")"
  response="$(curl -sS -X POST "$API_URL/api/faces/enroll" \
    -H "Content-Type: application/json" \
    -d "$(jq -n \
      --arg face_id "$face_id" \
      --arg customer_identifier "${RUN_ID}_${suffix}" \
      --arg customer_name "$customer_name" \
      '{face_id: $face_id, customer_identifier: $customer_identifier, customer_name: $customer_name}')")"

  echo "$response" | jq .
  customer_id="$(echo "$response" | jq -r '.customer_id')"
  ENROLLED_CUSTOMER_IDS+=("$customer_id")
}

assert_positive() {
  local image_path="$1"
  local suffix="$2"
  local expected_identifier="${RUN_ID}_${suffix}"
  local face_id
  local result
  local top_identifier

  face_id="$(detect_face "$image_path" "${RUN_ID}_${suffix}_positive")"
  result="$(recognize_face "$face_id")"
  echo "$result" | jq .

  top_identifier="$(echo "$result" | jq -r '.[0].customer_identifier // empty')"
  if [ "$top_identifier" != "$expected_identifier" ]; then
    echo "FAIL: expected $image_path to match $expected_identifier, got '${top_identifier:-no match}'"
    exit 1
  fi
}

assert_negative() {
  local image_path="$1"
  local identifier="$2"
  local face_id
  local result
  local count

  face_id="$(detect_face "$image_path" "$identifier")"
  result="$(recognize_face "$face_id")"
  echo "$result" | jq .

  count="$(echo "$result" | jq 'length')"
  if [ "$count" -ne 0 ]; then
    echo "FAIL: expected $image_path to return no matches, got $count"
    exit 1
  fi
}

echo "API: $API_URL"
echo "Minimum recognition confidence: $MIN_CONFIDENCE"
echo ""

echo "Cleaning previous example enrollments..."
cleanup_examples
echo ""

echo "1. Enrolling curated examples..."
enroll_customer "$ELON_ENROLL_IMAGE" "ELON" "Recognition Example - Elon Musk"
enroll_customer "$JENSEN_ENROLL_IMAGE" "JENSEN" "Recognition Example - Jensen Huang"

echo ""
echo "2. Positive checks: different same-person images should match..."
assert_positive "$ELON_POSITIVE_IMAGE" "ELON"
assert_positive "$JENSEN_POSITIVE_IMAGE" "JENSEN"

echo ""
echo "3. Negative check: Xi Jinping image should not match enrolled Elon or Jensen..."
assert_negative "$NEGATIVE_IMAGE" "${RUN_ID}_XI_negative"

echo ""
echo "PASS: curated positive and negative recognition examples behaved as expected."
