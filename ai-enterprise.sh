#!/bin/sh

if [ -z "$1" ]; then
  printf '%s\n' "Please pass ID of Project to be analyzed"
  exit 1
fi

if ! which jq >/dev/null 2>&1; then
ERRORS="'jq' is required
$ERRORS"
fi
if ! which mdcat >/dev/null 2>&1; then
ERRORS="'mdcat' is required
$ERRORS"
fi
if ! which mdcat >/dev/null 2>&1; then
ERRORS="'gcloud' is required
$ERRORS"
fi

if [ -n "$ERRORS" ]; then
  printf '%s\n' "Errors:" "$ERRORS"
  exit
fi

PROJECT_ID="prj-crtd-d"
LOCATION="us-central1"
MODEL_ID="gemini-2.5-flash"
RECENT_LINE_COUNT=500
REPORT_ID="report_${1}.txt"
FILTERED_REPORT="report_${1}_compact.txt"
GOOGLE_CLOUD_PROJECT="$1" ./detective > "$HOME/workspace/ai/chunker/chunks/$REPORT_ID"

grep -Ei "error|fail|exception|critical|fatal|500|501|502|503|400|401|403|404|429|denied" \
  "$HOME/workspace/ai/chunker/chunks/$REPORT_ID" | sort | uniq -c | sort -nr | \
  head -n "$RECENT_LINE_COUNT" > "$HOME/workspace/ai/chunker/chunks/$FILTERED_REPORT"

gsutil cp "$HOME/workspace/ai/chunker/chunks/$FILTERED_REPORT" "gs://prj-crtd-d-vertex-data/logs/$REPORT_ID"

curl --progress-bar "https://$LOCATION-aiplatform.googleapis.com/v1/projects/$PROJECT_ID/locations/$LOCATION/publishers/google/models/$MODEL_ID:streamGenerateContent" \
--header "Authorization: Bearer $(gcloud auth print-access-token --impersonate-service-account sa-vertex@prj-crtd-d.iam.gserviceaccount.com 2>/dev/null)" \
--header "Content-Type: application/json" \
--data-raw '{
  "contents": [{
    "role": "user",
    "parts": [
      { "text": "Please analyze these logs" },
      { "file_data": { "mime_type": "text/plain", "file_uri": "gs://prj-crtd-d-vertex-data/logs/'"$REPORT_ID"'" } }
    ]
  }]
}' > /tmp/response.json

if jq < /tmp/response.json >/dev/null 2>&1; then
  jq -r ".[].candidates[].content.parts[].text" < /tmp/response.json > /tmp/response.md
  mdcat /tmp/response.md
else
  printf '%s\n' "JSON format error" "" "Input:"
  cat /tmp/response.json
fi