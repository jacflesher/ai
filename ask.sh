#!/bin/sh

[ -z "$1" ] && ERRORS="$ERRORS
Please pass in a question in double-quotes."
if ! command -v jq >/dev/null 2>&1; then
  ERRORS="$ERRORS
The 'jq' is required to run this script."
fi
if ! command -v mdcat >/dev/null 2>&1; then
  ERRORS="$ERRORS
The 'mdcat' is required to run this script."
fi
[ -n "$ERRORS" ] && { printf 'Failed with errors: %s' "$ERRORS" >&2; exit 2; }

RESULT=$(
  printf '{"question": "%s", "threshold": 0.55}' "$1" | \
  curl "http://localhost:5000/ask" \
   --location --silent \
   --write-out "%{http_code}" \
   --output "/tmp/ask.json" \
   --header "Content-Type: application/json" \
   --data @-
)

if [ "$RESULT" -eq 200 ]; then
  jq -r ".answer" < "/tmp/ask.json" > "/tmp/answer.md"
  mdcat "/tmp/answer.md"
fi


