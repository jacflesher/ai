## CloudRun - Canary Monitoring

   Dynatrace canary monitoring is enabled for Cloud Run in preprod and prod regions. If performance degradation is detected, the system will alert the Serverless Monitoring Alerts channel in Microsoft Teams and create an incident ticket.


### Canary flow diagram

![](../img/cr-flowdiagram.png)

### Master GET

   Master “GET” canary uses a logic that fetches information about the service via an API and then performs checks based on the response's status code.

-	404 (Not Found): The service doesn't exist. The test skips further checks.
-	200 (OK) and Status is "True": The service is running. The code checks if the service has been running for longer than expected (implying it should have been deleted). If so, the test fails.
-	200 (OK) and Status is "Unknown": The service is in a transitional state (e.g., deploying). The code checks if the deployment has taken too long. If so, the test fails.
-	200 (OK) and Status is "False": The service has failed to deploy. The test fails.


     The code uses timestamps to measure the time the service has been running or the duration of the deployment process. It uses the api object (presumably provided by a testing framework) to log information, skip requests, and report failures.


HTTP URL :

(Example for central1 region)

v1: https://us-central1-run.googleapis.com/apis/serving.knative.dev/v1/namespaces/ford-4dbd7038019cc45cc32a5084/services/cloudrun-monitoring-app-create-central1-pp

v2: https://us-central1-run.googleapis.com/v2/projects/ford-4dbd7038019cc45cc32a5084/locations/us-central1/services/cloudrun-monitoring-path-appv2-central1-pp



### How to navigate in dynatrace

Prod : https://www.dynatrace.ford.com/

qa   : https://wwwqa.dynatrace.ford.com/

Application Observability -> Synthetic -> search for “GCP Cloud Run”


![](../img/Dynatrace.png)

Select the monitor and click on "View settings"

![](../img/view-settings.png)

Click on "HTTP Requests" to view all the requests that been configured.

![](../img/HTTP-Requests.png)

## Preprod Canaries

|Monitoring Name	|Frequency	|First Status Check	|Update Service	|Get status after update	|Delete Test	|Get status after deletion	|Create Test|
|-----------------|-----------|--------------------|-----------------|--------------------------|--------------|--------------------------|-----------|
|[GCP Cloud Run - Asia South1 - Internal API Monitoring - v1 Preprod](https://wwwqa.dynatrace.ford.com/e/114d327e-ea9d-46cc-92d3-3967eaedacde/ui/http-monitor/HTTP_CHECK-68DBA96E142A0959?gtf=-10m&gf=all)	|5 Min	|Yes	|No  |No  |Yes	|Yes	|Yes|
|[GCP Cloud Run - Asia South1 - Internal API Monitoring - v2 Preprod](https://wwwqa.dynatrace.ford.com/e/114d327e-ea9d-46cc-92d3-3967eaedacde/ui/http-monitor/HTTP_CHECK-56A510D3D3208201?gtf=-10m&gf=all)	|5 Min	|Yes	|No  |No  |Yes	|Yes	|Yes|
|[GCP Cloud Run - Asia SouthEast1 - Internal API Monitoring - v1 Preprod](https://wwwqa.dynatrace.ford.com/e/114d327e-ea9d-46cc-92d3-3967eaedacde/ui/http-monitor/HTTP_CHECK-58F1AC97A187C0C5?gtf=-10m&gf=all)	|5 Min	|Yes	|No  |No  |Yes	|Yes	|Yes|
|[GCP Cloud Run - Asia SouthEast1 - Internal API Monitoring - v2 Preprod](https://wwwqa.dynatrace.ford.com/e/114d327e-ea9d-46cc-92d3-3967eaedacde/ui/http-monitor/HTTP_CHECK-83FB31300CBC4B03?gtf=-10m&gf=all)	|5 Min	|Yes	|No  |No  |Yes	|Yes	|Yes|
|[GCP Cloud Run - Europe West2 - Internal API Monitoring - v1 Preprod](https://wwwqa.dynatrace.ford.com/e/114d327e-ea9d-46cc-92d3-3967eaedacde/ui/http-monitor/HTTP_CHECK-1ADBB86004E74A00?gtf=-10m&gf=all)	|5 Min	|Yes	|No  |No  |Yes	|Yes	|Yes|
|[GCP Cloud Run - Europe West2 - Internal API Monitoring - v2 Preprod](https://wwwqa.dynatrace.ford.com/e/114d327e-ea9d-46cc-92d3-3967eaedacde/ui/http-monitor/HTTP_CHECK-BAE195C0B984DD4B?gtf=-10m&gf=all)	|5 Min	|Yes	|No  |No  |Yes	|Yes	|Yes|
|[GCP Cloud Run - Europe West3 - Internal API Monitoring - v1 Preprod](https://wwwqa.dynatrace.ford.com/e/114d327e-ea9d-46cc-92d3-3967eaedacde/ui/http-monitor/HTTP_CHECK-055E15B6B056EC25?gtf=-10m&gf=all)	|5 Min	|Yes	|No  |No  |Yes	|Yes	|Yes|

## Prod Canaries

|Monitoring Name	|Frequency	|First Status Check	|Update Service	|Get status after update	|Delete Test	|Get status after deletion	|Create Test|
|-----------------|-----------|--------------------|-----------------|--------------------------|--------------|--------------------------|-----------|
|[GCP Services Cloud Run - Asia South1- Internal - v2 - Holding](https://www.dynatrace.ford.com/e/8436fe71-6539-4ea3-aab8-a9985ae713d4/ui/http-monitor/HTTP_CHECK-D2A8D3DC80B65A28?gtf=-30d%20to%20now&gf=all)	|2 Min	|Yes	|Yes	|Yes	|Yes	|Yes	|Yes|
|[GCP Services Cloud Run - Asia South2 -Internal - v2 - Holding](https://www.dynatrace.ford.com/e/8436fe71-6539-4ea3-aab8-a9985ae713d4/ui/http-monitor/HTTP_CHECK-C47B74A6074AC584?gtf=-30d%20to%20now&gf=all)	|2 Min	|Yes	|Yes	|Yes	|Yes	|Yes	|Yes|
|[GCP Services Cloud Run - Asia SouthEast1 - Internal - v2 - Holding](https://www.dynatrace.ford.com/e/8436fe71-6539-4ea3-aab8-a9985ae713d4/ui/http-monitor/HTTP_CHECK-6F307295111BBC4D?gtf=-30d%20to%20now&gf=all)	|2 Min	|Yes	|Yes	|Yes	|Yes	|Yes	|Yes|
|[GCP Services Cloud Run - Europe West2 - Internal - v2 - Holding](https://www.dynatrace.ford.com/e/8436fe71-6539-4ea3-aab8-a9985ae713d4/ui/http-monitor/HTTP_CHECK-88BD2D6BBFFE7391?gtf=-30d%20to%20now&gf=all)  |2 Min	|Yes  |Yes	|Yes	|Yes	|Yes	|Yes|
|[GCP Services Cloud Run - Europe West3 - Internal - v2 - Holding](https://www.dynatrace.ford.com/e/8436fe71-6539-4ea3-aab8-a9985ae713d4/ui/http-monitor/HTTP_CHECK-2A3014B6AB55E2D7?gtf=-30d%20to%20now&gf=all)	|2 Min	|Yes	|Yes	|Yes	|Yes	|Yes	|Yes|
|[GCP Services Cloud Run - US Central1 - Internal - v2 - Holding](https://www.dynatrace.ford.com/e/8436fe71-6539-4ea3-aab8-a9985ae713d4/ui/http-monitor/HTTP_CHECK-7CD811D747C09306?gtf=-30d%20to%20now&gf=all)	|2 Min	|Yes	|Yes	|Yes	|Yes	|Yes	|Yes|
|[GCP Services Cloud Run - US East4 - Internal - v2 - Holding](https://www.dynatrace.ford.com/e/8436fe71-6539-4ea3-aab8-a9985ae713d4/ui/http-monitor/HTTP_CHECK-6E6FE9B12CC38C4E?gtf=-30d%20to%20now&gf=all)	|2 Min	|Yes	|Yes	|Yes	|Yes	|Yes	|Yes|


## Backup of Post-execution script used in the HTTP Requests

GCP CloudRun First Status(Request order 5)

[Post-execution script](../canary-monitoring/postscripts/first-status-post-script.js)

GCP Update status(Request order 6)

[Request body-RAW](../canary-monitoring/postscripts/update-status-request-body.js)

GCP Cloud Run Get Status After Update(Request order 7)

[Post-execution script](../canary-monitoring/postscripts/Get-Status-After-Update.js)

GCP Cloud Run Get Status After Deletion
(Request order 9)

[Post-execution script](../canary-monitoring/postscripts/Get-Status-After-Deletion.js)

GCP CloudRun create (Request order 10)

[Request body](../canary-monitoring/postscripts/CloudRun-create.js)







