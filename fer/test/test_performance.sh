#!/bin/bash

NUM_REQUESTS=10

START_TOTAL=$(date +%s%3N)

for ((i=1;i<=NUM_REQUESTS;i++)); do
  curl -X 'POST' \
  'http://localhost:8000/recognize/file' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@test_image.png'
done

wait

END_TOTAL=$(date +%s%3N)

DIFF_TOTAL=$((END_TOTAL - START_TOTAL))
AVG_TIME_MILLI=$((DIFF_TOTAL / NUM_REQUESTS))

echo
echo "Total time taken for $NUM_REQUESTS requests: $((DIFF_TOTAL / 1000)) seconds"
echo "Average time per request: $AVG_TIME_MILLI milliseconds"
