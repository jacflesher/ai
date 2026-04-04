# Cloud Run Ops Dashboard
This dashboard provides a clear, immediate view into how our Cloud Run services are performing . It's built to help us quickly spot problems and keep applications running smoothly by focusing on the key metrics.

## Access the Dashboard Here:
https://vpx74458.apps.dynatrace.com/ui/apps/dynatrace.dashboards/dashboard/717b2823-3260-4590-aef3-50318cde1edb#from=now%28%29-30m&to=now%28%29&vfilter_Region=3420b2ac-f1cf-4b24-b62d-61ba1ba8ed05*&vfilter_Interval=10+m


## What We Monitor (Key Dashboard Metrics):

### Synthetic service health
- This section displays the overall health and availability of our services across all 7 regions. It's calculated using our synthetic monitors, which are set up in each of these regions to constantly check how our services are performing . This gives us crucial insights into region wise service availability.

### Gateway & Network Errors (502, 503, 504) and Service Uptime:
- This tracks problems that happen before requests even reach our Cloud Run apps, often related to network infrastructure or load balancers. 502 Bad Gateway, 503 Service Unavailable, and 504 Gateway Timeout indicate these upstream issues. We also calculate the overall availability of these gateway and Network related errors.

### Application Errors (500) and App Availability:
- This focuses on errors that occur inside our Cloud Run applications themselves, meaning the app's code encountered an unexpected issue. A 500 Internal Server Error is a common sign of this. This metric also calculates the overall availability, indicating if it's online and ready to serve requests.

### Client Request Errors (4xx) and Service Availability:
- This metrics specifically tracks issues where the problem originates from the client's request itself, such as an invalid input or insufficient permissions. We concentrate on specific errors like 401 Unauthorized (authentication failed), 403 Forbidden(lacks necessary permissions), 404 Not Found (requested resource doesn't exist), 409 Conflict (request conflicts with current server state), and 429 Too Many Requests (client exceeding rate limits). We also monitor the service's overall availability to handle these types of client interactions.

### Internal Network Performance (VPC Connector Throughput & CPU):
- This shows how much data is flowing through our private network connections (VPC Connectors) and how busy their processors are. These connections are crucial for Cloud Run apps to securely talk to other internal Ford systems, ensuring fast and reliable communication.

### Top 25 Apps with Server Problems (5xx Errors, excluding 4xx):
- This list highlights the twenty-five Cloud Run applications that are experiencing the most server-side issues (any 5xx Server Error). By showing this, we can quickly identify which apps are the most unstable and need immediate attention from their development teams.

### Apigee basepath error with cloud run as backend:
- This metrics shows us the list of apigee base path errors with the cloud run as backend.
This metric specifically identifies problems where Apigee API Gateway is failing to connect to, or get a proper response from, a Cloud Run service it's meant to manage. It helps us troubleshoot communication breakdowns between our API management layer and our backend services.

