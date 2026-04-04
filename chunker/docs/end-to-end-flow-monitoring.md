## Creating monitoring dashboards for eventarc

To create the montioring dashboard for eventarc, we need to work with the monitoring team to set the initial metrics. There three types of dashboard for all the services which are `synthetic dashboard`, `SRE dashboard` and `SLI/SLO dashboard`.

### Steps involved:
- The synthetic dashboard will be created first with demo eventarc service triggers. 
- Once the synthetic dashboard is created, from serverless team we are supposed to create the eventarc canaries in pre-prod and prod project allocated for eventarc

## How to create the canaries:

Canaries are eventarc that deployed in all the ford specific regions and used by the monitoring team to monitor the service health. 

### Steps involved:
- The eventarc service is an event based trigger so select the destination service to use for the canaries. 
- [Here](https://github.ford.com/serverless/eventarc-cd) we are using the cloud run as destination service. 
- The trigger type for eventarc is `PubSub message published` so we need to create the pubsub topic as well. 
- Create the terraform template from any canaries to create the eventarc, pubsub and cloud run for monitoring.
- The eventarc cd has a managed tekton pipeline which will be used for deployment in prod followed by pre-prod.
- once deployed please check with the montioring team to configure the synthetic dashboard and its alerts. 
- once the synthetic dashboard is pointing towards the correct eventarc then follow up for SRE dashboards. 

### Reference:

- Dynatrace link for eventarc monitoring - [here](https://wwwqa.dynatrace.ford.com/e/114d327e-ea9d-46cc-92d3-3967eaedacde/#dashboard;gtf=-72h%20to%20now;gf=-7339779436848507076;id=69ef602e-2032-45d4-ad07-9e1d00074b7f)



