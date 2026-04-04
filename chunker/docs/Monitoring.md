# Monitoring # 

- As service team we have to make sure that our service is available in each and every region so that users will get smooth experience. 
- Currently there are seven canaries deployed in "Pre-prod" and "Prod" projects in Cloud Run and Cloud Functions.
-  The canaries will check the availability of Cloud Run and Cloud Functions in each and every region. The regions which is being monitored are as follows:

- There are no thresholds for Canaries. If the canaries are down for more than 5 mins it is considered as a downtime.

1. us-cental1
2. us-east4
3. asia-south1
4. asia-south2
5. asia-southeast1
6. europe-west2
7. europe-west3

## Cloud Run ##

Canaries are deployed in below projects : [Cloud Run Dynatrace Dashboard Link](https://wwwqa.dynatrace.ford.com/e/114d327e-ea9d-46cc-92d3-3967eaedacde/#dashboard;gtf=-20m%20to%20now;gf=-7339779436848507076;id=46d6a330-d936-46b0-b29d-ee0d483f5e67)
  **a. Pre-prod** = **ford-4dbd7038019cc45cc32a5084**
  **b. Prod** = **ford-bf4f3f52c6c80b800ea99472**

- Below are the Canaries set for Cloud Run in "Pre-Prod" and "Prod".

![Flow Chart](img/canarycrpp.png)

![Flow Chart](img/canarycr.png)

[Cloud Run Dynatrace Dashboard for External Reachability via Apigee](https://wwwqa.dynatrace.ford.com/e/114d327e-ea9d-46cc-92d3-3967eaedacde/#dashboard;id=692adddb-c22c-4b0e-bd45-fae334cab9d2;gtf=-10m;gf=-771801749622999336)

## Cloud Functions ##

Canaries are deployed in below projects : [Cloud Function Dynatrace Dashboard Link](https://wwwqa.dynatrace.ford.com/e/114d327e-ea9d-46cc-92d3-3967eaedacde/#dashboard;gtf=-20m%20to%20now;gf=-7339779436848507076;id=b00ef83d-ea83-456e-a4e6-7c6fdd46c574)
  **a. Pre-prod** = **ford-8ed30880b2e4d762e9ffe33a**
  **b. Prod** = **ford-906495ce4c398e670f27e85b**

- Below are the Canaries set for Cloud Functions in "Pre-Prod" and "Prod".

![Flow Chart](img/cfpp.png)

![Flow Chart](img/cfprod.png)

- Active instances by Function are represented within this dashboard.

![Flow Chart](img/cfregion.png)

- Click on "Configure in Data Explorer". Once you navigate to this page, you will be able to see more drilled-down view of the data.

- Performance and Memory usage are represented within this dashboard.

- Click on "Configure in Data Explorer". Once you navigate to this page, you will be able to see more drilled-down view of the data.


![Flow Chart](img/cfactive.png)

- Above graph will show you more granular information about Cloud Functions.

- Click on "Configure in Data Explorer". Once you navigate to this page, you will be able to see more drilled-down view of the data.


## How to monitor : ##

- If the canary goes down in any of the seven region for more than five minutes it will generate an alert .

- We have one dedicated webex space for monitoring alerts "GCP Serverless Monitoring Alerts". In this space the message will get generated and also the BMC Incident Ticket gets generated.

  
  ![Flow Chart](img/message.png)


- If you click on "Open in Browser", it will take you to the page where the incident took place.


- Once the issue is resolved, another message will get generated in the webex space "GCP Serverless Monitoring Alerts".

   ![Flow Chart](img/message2.png)

### CURD canary monitor - [Referrence Link](https://github.com/ford-cloud/serverless-docs/blob/main/operations-guide/monitoring-and-sre-dashboards/canary-monitoring/cloudRun-canary-monitoring(Dynatrace).md)
