

#### Environment Variables
```sh
PROJECT_ID="prj-crtd-d"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format "value(projectNumber)")
LOCATION="us-central1"
BUCKET_NAME="${PROJECT_ID}-vertex-data"
```

#### Create the bucket
```sh
gcloud storage buckets create "gs://$BUCKET_NAME" \
    --project="$PROJECT_ID" \
    --location="$LOCATION" \
    --uniform-bucket-level-access
```

#### Grant your impersonated Service Account access
```sh
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:sa-vertex@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"
gcloud storage buckets add-iam-policy-binding "gs://prj-crtd-d-vertex-data" \
    --member="serviceAccount:sa-vertex@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

```

#### Grant the Vertex AI Service Agent access (This is the one people often miss!)
```sh
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:service-$PROJECT_NUMBER@gcp-sa-aiplatform.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"
gcloud storage buckets add-iam-policy-binding "gs://prj-crtd-d-vertex-data" \
    --member="serviceAccount:service-$PROJECT_NUMBER@gcp-sa-aiplatform.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

```



gsutil cp "$HOME/workspace/ai/chunker/chunks/report.txt" gs://prj-crtd-d-vertex-data/logs/report.txt

