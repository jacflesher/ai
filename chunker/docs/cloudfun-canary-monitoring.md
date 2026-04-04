# Documentation: Cloud Functions Canary Monitoring

This document describes about our Cloud Functions Canary monitoring, designed for proactive monitoring and end-to-end testing of our Cloud Functions.

## 1. Architecture Overview

*   **Google Cloud Scheduler:** Triggers canary tests at set intervals.
*   **Google Cloud Run:** Hosts the canary logic, executing the tests.
*   **Google Cloud Functions:** The services under test.
*   **Redis:** Stores temporary error states (TTL = 1 hour) to prevent alert storms and trigger ticketing for persistent issues.

### Redis Error Handling Logic:
When an error is detected by a canary:
1.  The error details are stored in Redis with a 1-hour Time-To-Live (TTL).
2.  Before storing, the system checks if the same error for the same function is already present in Redis.
3.  If the error is not found in Redis (first occurrence), it's stored.
4.  If the error is found in Redis (second consecutive occurrence within 1 hour), a ticket is automatically submitted to our group.

---


## 2. Canary Flow 2: End-to-End Function Lifecycle (`cf canary create-delete`)

This canary is designed to validate the entire lifecycle of a Cloud Function: checking its status, creating a new one, verifying its creation, and then deleting it. This ensures that the underlying Cloud Functions API and infrastructure are fully operational for all key operations. Performs full end-to-end test of Cloud Function management (GET, CREATE, DELETE).

**Sequence:** `GET` → `CREATE` → `GET` → `DELETE`

### Flow Description:

1.  **Scheduler Trigger:** Google Cloud Scheduler triggers by sending a request to its dedicated Cloud Run endpoint.Scheduler triggers for every 7 minutes.

2.  **Initial GET Request:** The Cloud Run service starts by performing a `GET` request to check the status of a Cloud Function on which action is being performed.

3.  **Function Status Evaluation (Initial):** The canary checks the function's initial status:
    *   **If Function Not Available:** The function doesn't exist. The canary then continues the sequence to create a new function.
    *   **If Function is Not Deleted Properly (from a previous run):** If a previous run of this canary failed to clean up, this step identifies it. This is considered an error and logs error in Redis . The canary then proceeds to continue the sequence.
    *   **If Function is Available in a Failed State:** The function exists but is not working. The canary records this as an error and then proceeds to continue the sequence.
    *   **If Function is Available & Deploying/Updating:** The function is in a transitional state and taking longer than given time interval ,logs an error and continue the sequence


4.  **Continue the Sequence (Core Operations):** After the initial status evaluation, the canary performs the following operations:
    *   **CREATE Function:** The canary attempts to create a new temporary Cloud Function. This tests the creation API.
        *   **Error Handling:** If `CREATE` fails: The error is logged and if threshold exceeds, a ticket is submitted.
    *   **GET New Function:** Once created, it performs a `GET` request to verify the newly created function is available and active. This confirms successful deployment.
        *   **Error Handling:** If this `GET` fails: The error is logged and if threshold exceeds, a ticket is submitted.
    *   **DELETE Function:** Finally, the canary deletes the temporary Cloud Function it just created. This ensures cleanup and tests the deletion API.
        *   **Error Handling:** If `DELETE` fails: The error is logged and if threshold exceeds, a ticket is submitted.

### Flowchart:

### Canary create-delete Flow diagram

![](../img/cfcanarycreate-delete.png)

### Deployment Details:

*   **Cloud Run Services Details:**
 
    *   `Project`    : `prj-cloud-functions-pp`
    *   `us-central1`: [cf-canary-create-delete-uscentral1-pp](https://console.cloud.google.com/run?inv=1&invt=Ab2XLA&project=ford-8ed30880b2e4d762e9ffe33a#:~:text=cf%2Dcanary%2Dcreate%2Ddelete%2Duscentral1%2Dpp)
    *   `us-east4`   : [cf-canary-create-delete-useast4-pp](https://console.cloud.google.com/run?inv=1&invt=Ab2XLA&project=ford-8ed30880b2e4d762e9ffe33a#:~:text=cf%2Dcanary%2Dcreate%2Ddelete%2Duseast4%2Dpp)
    *   `europe-west3`: [cf-canary-create-delete-europe-west3-pp](https://console.cloud.google.com/run?inv=1&invt=Ab2XLA&project=ford-8ed30880b2e4d762e9ffe33a#:~:text=cf%2Dcanary%2Dcreate%2Ddelete%2Deurope%2Dwest3%2Dpp)
    *   `europe-wes2`: [cf-canary-create-delete-europe-west2-pp](https://console.cloud.google.com/run?inv=1&invt=Ab2XLA&project=ford-8ed30880b2e4d762e9ffe33a#:~:text=cf%2Dcanary%2Dcreate%2Ddelete%2Deurope%2Dwest2%2Dpp)
    
    *   `Project`    : `prj-cloud-functions-p`
    *   `us-central1`: [cf-canary-create-delete-uscentral1-prod](https://console.cloud.google.com/run?inv=1&invt=Ab2XLA&project=ford-906495ce4c398e670f27e85b&rapt=AEjHL4M0x13Sh-vagCnR8Vkkdim-jZDtzBebM6ME2H0Jb2VSH3h_Oku_UOxmnmak-lUxtnBWZvXvW0xuUQqT5wzAfyjWFln4DbBGISlnb_Qf-JcWvRnquJo#:~:text=cf%2Dcanary%2Dcreate%2Ddelete%2Duscentral1%2Dprod)
    *   `us-east4`   : [cf-canary-create-delete-useast4-prod](https://console.cloud.google.com/run?inv=1&invt=Ab2XLA&project=ford-906495ce4c398e670f27e85b&rapt=AEjHL4M0x13Sh-vagCnR8Vkkdim-jZDtzBebM6ME2H0Jb2VSH3h_Oku_UOxmnmak-lUxtnBWZvXvW0xuUQqT5wzAfyjWFln4DbBGISlnb_Qf-JcWvRnquJo#:~:text=cf%2Dcanary%2Dcreate%2Ddelete%2Duseast4%2Dprod)
    *   `europe-west3`: [cf-canary-create-delete-europe-west3-prod](https://console.cloud.google.com/run?inv=1&invt=Ab2XLA&project=ford-906495ce4c398e670f27e85b&rapt=AEjHL4M0x13Sh-vagCnR8Vkkdim-jZDtzBebM6ME2H0Jb2VSH3h_Oku_UOxmnmak-lUxtnBWZvXvW0xuUQqT5wzAfyjWFln4DbBGISlnb_Qf-JcWvRnquJo#:~:text=cf%2Dcanary%2Dcreate%2Ddelete%2Deurope%2Dwest3%2Dprod)
    *   `europe-wes2`: [cf-canary-create-delete-europe-west2-prod](https://console.cloud.google.com/run?inv=1&invt=Ab2XLA&project=ford-906495ce4c398e670f27e85b&rapt=AEjHL4M0x13Sh-vagCnR8Vkkdim-jZDtzBebM6ME2H0Jb2VSH3h_Oku_UOxmnmak-lUxtnBWZvXvW0xuUQqT5wzAfyjWFln4DbBGISlnb_Qf-JcWvRnquJo#:~:text=cf%2Dcanary%2Dcreate%2Ddelete%2Deurope%2Dwest2%2Dprod)


### Sample incident ticket -Priority(MEDIUM)

![](../img/Sample_creation_error_ticket.png)

## 4. Troubleshooting & Alerts

### Alerting Mechanism:
*   **Incident Tickets:** Errors trigger incident tickets via ServiceNow after a threshold of 2 occurrences within 1 hour in Redis.
*   **Cloud Logging:** Detailed canary logs (successes, warnings, errors) are available in Google Cloud Logging for debugging.

### Common Alert Triggers & Troubleshooting:

When an incident ticket is created , we can investigate the following accordingly:

1.  **Function Not Found:**
    *   **Meaning:** The target Cloud Function could not be located.
    *   **Troubleshooting:**
        *   Verify correct function name and region.
        *   Check service account permissions (e.g., `cloudfunctions.functions.get`).

2.  **Function in Failed/Problematic State:**
    *   **Meaning:** Function is `FAILED` or stuck in `DEPLOYING`/`UPDATING`/`DELETING` for too long (e.g., >5min).
    *   **Troubleshooting:**
        *   Check Cloud Function console for its state, logs, and deployment history.
        *   Look for recent changes or configuration issues.

3.  **Function Operation Timeout:**
    *   **Meaning:** A Cloud Function operation (create, update, delete) did not complete within its time limit.
    *   **Troubleshooting:**
        *   Review Cloud Function's operation history and logs for reasons for slowness.
        *   Check for large deployment package sizes.

4.  **API Errors:**
    *   **Meaning:** The canary's Cloud Run service failed to communicate with the Cloud Functions API.
    *   **Troubleshooting:**
        *   Verify Cloud Run service account has all necessary API permissions.
        *   Check Google Cloud status dashboard for outages.

5.  **Redis Connection/Threshold Errors:**
    *   **Meaning:** Canary cannot connect to Redis, or there's an issue with error counting.
    *   **Troubleshooting:**
        *   Check Redis instance health and network connectivity from Cloud Run.
        *   Verify `REDIS_CONFIG` environment variable for correct details.

6.  **General/Unexpected Errors:**
    *   **Meaning:** A broad exception occurred in the canary's logic.
    *   **Troubleshooting:**
        *   Verify the full traceback in Cloud Logging for the specific error details.

---





