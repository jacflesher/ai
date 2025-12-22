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
  jq -n --arg q "$1" --arg s "$(whoami)" --argjson t 0.55 \
  '{question: $q, session_id: $s, threshold: $t}' | \
  curl "http://localhost:8080/ask" \
   --location --silent \
   --write-out "%{http_code}" \
   --output "/tmp/ask.json" \
   --header "Content-Type: application/json" \
   --data @-
)

if [ "$RESULT" -eq 200 ]; then
  jq -r ".answer" < "/tmp/ask.json" > "/tmp/answer.md"
  mdcat "/tmp/answer.md"
else
  printf '\nError!!!\n%s\n\n' "$RESULT"
  cat "/tmp/ask.json"
fi
