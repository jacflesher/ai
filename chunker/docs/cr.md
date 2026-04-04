# SRE Dashboards #

SRE stands for Service Reliability Engineering.
SRE dashboards are designed to monitor the health of the services, helps team to provide more visibility on how their services are working, to provide smooth experience to users. Our Dashboards covers Four Golden signals as following:

## SRE Cloud Run ##

SRE Dashboards are useful to track outages, helps to detect the Latency. In Ford Cloud Run dashboard covers following "Golden Signals":

**1**. **Latency**
**2**. **Error Class: 4xx, 5xx**
**3**. **Traffic request count**
**4**. **Container memory utilization**

## 1. Latency ##

- Latency is the delay in network communication for transferring the data. The more the latency the more delay user will face to get the output. In Ford we use GCP (Google Cloud Platform) if there is an issue in Google side or Ford side internet it will create latency. 

![Flow Chart](images/lat1.png)

- Above image will show you the the latencies caused in Cloud Run.

- X-axis represents "Date".

- Y-axis represents "Time" (for how long the latency was there) .

- Latency depends on the bandwidth of the Ford and Google Network.

- For more information you can click on right side drop down and then click on "Configure data in explorer".

![Flow Chart](images/lat2.png)


- There are queries set based on which you can filter it based on Project-id, GCP region. It will show when and at what time the latency was caused. You will get the exact time,date and for how long latency was there.

- You can split it and then click on "Run Query"

- It will display the projects in which latency was caused along with the region.

- X-axis represents  "Date".

- Y-axis represents "Time".

Note: If Latency goes above 60000 ms and if the latency continues for more than 5 mninutes it will generate an alert.

## 2. Error Class ##

-  Error is a  condition when the receiver's information does not match the sender's information. There are many error class, in our dashboards we have 4xx and 5xx tile configured. This tile is useful to provide the information if the error is caused by app team or something is wrong with the server.

Note: If Error Count goes above 2000 Count, it will generate alert.

### a. 4XX ###

- 4XX error indicates that the requested resource or web page could not be found on the server. Other examples of 4xx errors include “401 Unauthorized”, “403 Forbidden”, and “400 Bad Request” and many more. 

- 4XX tile is useful to keep eye on the applications and to analyzed the issues which will cause due to errors in application.

![Flow Chart](images/4xx1.png)


- Click on "Configure Data in Explorer".

![Flow Chart](images/4xx.png)

- Filter it by "response_code", it will display the projects which has caused the error which you have provided in response code.

- Select "gcp_project_id"

- X-axis represents the "Threshold".

- Y-axis represents the "Request Count".

- Results will get displayed based on the filters selected by you.




### b. 5xx ###

- 5XX error indicates server related or API is down,indicates in which project the service is down, service unavailability and many more.


- 5XX tile is useful to detect issues which will get cause due to timeout, service not available etc.

![Flow Chart](images/5xx2.png)

-  Click on "Configure Data in Explorer"

![Flow Chart](images/5xx.png)

- Filter it by "response_code", it will display the projects which has caused the error which you have provided in response code.

- Select "gcp_project_id"

- X-axis  represent the threshold.

- Y-axis  represent the request count.

- Results will get displayed based on the filters selected by you.


## 3. Traffic Request Count ##

- Traffic request count is the number of request called by user. It will show in which project the url was called and user made the request.

- The number of HTTP requests hitting the Cloud Run services that are deployed.


![Flow Chart](images/traffic1.png)

-  Click on "Configure Data in Explorer"


![Flow Chart](images/traffic2.png)

- Split it based on location, gcp_project_id. It will display the number of request based on project.

- X-axis  represent the "Request Count".

- Y-axis represent the "Threshold".

Note: If the Traffic Request Count goes below 1000 KB it will generate an alert.


## 4. Container Memory Utilization ##

- Container Memory Utilization will display the memory utlized by Cloud Run.


![Flow Chart](images/container2.png)

- It is usedful to track which service is consuming more memory, useful to track the memory usage by the services.

![Flow Chart](images/container.png)


- X-axis represents "Threshold".

- Y-axis represents "Container memory utilization".






