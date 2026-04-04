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

# PROJECT_ID="prj-crtd-d"
PROJECT_ID="ford-afd20ec4c8a9b3a7599c2ef8"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format "value(projectNumber)")
LOCATION="us-central1"
MODEL_ID="gemini-2.5-pro"
REPORT_ID="report_${1}.txt"

# gcloud logging read \
#   --project "${1}" \
#   --limit "5000" \
#   --freshness "30d" \
#   --format "value(textPayload)" | grep '\S' > "/tmp/$REPORT_ID"

gcloud logging read \
  --project "${1}" \
  --limit "500" \
  --freshness "30d" \
  --format json > "/tmp/$REPORT_ID"

# gcloud storage buckets create "gs://vertex-data-$PROJECT_NUMBER" \
#   --project "$PROJECT_ID" \
#   --location "us-central1"

gcloud storage cp "/tmp/$REPORT_ID" "gs://vertex-data-$PROJECT_NUMBER/logs/$REPORT_ID" --project "$PROJECT_ID"

curl --progress-bar "https://$LOCATION-aiplatform.googleapis.com/v1/projects/$PROJECT_ID/locations/$LOCATION/publishers/google/models/$MODEL_ID:streamGenerateContent" \
--header "Authorization: Bearer $(gcloud auth print-access-token 2>/dev/null)" \
--header "Content-Type: application/json" \
--data-raw '{
  "contents": [{
    "role": "user",
    "parts": [
      { "text": "Please analyze these GCP project logs" },
      { "file_data": { "mime_type": "text/plain", "file_uri": "gs://vertex-data-'"$PROJECT_NUMBER"'/logs/'"$REPORT_ID"'" } }
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