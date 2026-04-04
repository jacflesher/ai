## SRE Cloud Functions ##

In Ford SRE dashboard of Cloud Functions following "Golden Signals":

**1**. **Latency**
**2**. **Error Class: 4xx, 5xx**
**3**. **Traffic request count**
**4**. **Container memory utilization**

## 1. Latency ##

- Latency is the delay in network communication for transferring the data. The more the latency the more delay user will face to get the output. In Ford we use GCP (Google Cloud Platform) if there is an issue from Google side or Ford side internet it will create latency. 

- Above image will show you the the latencies caused in Cloud Run.

- X-axis represents "Date".

- Y-axis represents "Time" (for how long the latency was there) .

- Latency depends on the bandwidth of the Ford and Google Network.

- For more information you can click on right side drop down and then click on "Configure data in explorer".

![Flow Chart](images/cflat.png)


- There are queries set based on which you can filter it based on Project-id, GCP region. It will show when and at what time the latency was caused. You will get the exact time,date and for how long latency was there.

- You can split it and then click on "Run Query"

- It will display the projects in which latency was caused along with the region.

***Note: If Latency goes above 120 Seconds and if the latency continues for more than 5 minutes it will generate an alert.***

## 2. Error Class ##

-  Error is a  condition when the receiver's information does not match the sender's information. There are many error class, in our dashboards we have 4xx and 5xx tile configured. This tile is useful to provide the information if the error is caused by app team or something is wrong with the server.

![Flow Chart](images/cf4xx.png)

***Note: If Error Count goes above 200 count , it will generate an alert.***

### a. 4XX ###

- 4XX error indicates that the requested resource or web page could not be found on the server. Other examples of 4xx errors include “401 Unauthorized”, “403 Forbidden”, and “400 Bad Request” and many more. 

- 4XX tile is useful to keep eye on the applications and to analyzed the issues which will cause due to errors in application.


- Click on "Configure Data in Explorer".

- Filter it by "response_code", it will display the projects which has caused the error which you have provided in response code.

- Select "gcp_project_id"

- X-axis represents the "Threshold".

- Y-axis represents the "Request Count".

- Results will get displayed based on the filters selected by you.

![Flow Chart](images/cf4xx5xx.png)

### b. 5xx ###

- 5XX error indicates server related or API is down,indicates in which project the service is down, service unavailability and many more.


- 5XX tile is useful to detect issues which will get cause due to timeout, service not available etc.


- Filter it by "response_code", it will display the projects which has caused the error which you have provided in response code.

- Select "gcp_project_id"

- X-axis  represent the "Threshold".

- Y-axis  represent the "Request Count".

- Results will get displayed based on the filters selected by you.

## Traffic Execution Count ##

- Execution count is the number of time a request which is called. This tile will show how many times end-point of Cloud Function is being called.

- This tile will show the traffic which will generate when a particular end-point is being triggered.

![Flow Chart](images/cfexecution.png)

- X-axis  represent the "Threshold".

- Y-axis represents the "Request Count".

***Note: If the traffic request count goes below 1000 KB, it will generate an alert.***


## Saturation Response Size ##

- Saturation is high-level overview of response sizes. 

- Saturation will show how full your service is. It will generate an alert if the count goes above the threshold value.

![Flow Chart](images/cfsaturation.png)

***Note: If Saturation goes below 2048 bytes it will generate an alert.***

