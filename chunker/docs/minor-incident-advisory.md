# Serverless Minor Incident Advisory Process
This process is to be used when there is an issue impacting multiple applications in a production or non-production environment and it is deemed to have medium or low impact

To create a minor incident advisory, the following steps should be performed.

## Table of Contents
* [Part 1 Communication Prep](#part-1-communication-prep)
* [Part 2 Send the Initial Communication](#part-2-send-the-initial-communication)
* [Part 3 Send the Update Communication](#part-3-send-the-update-communication)
* [Part 4 Send the Final Communication](#part-4-send-the-final-communication)

[Back to Top](#serverless-minor-incident-advisory-process) | [Back to Main](README.md)

## Part 1 Communication Prep
1. __Prepare a problem statement__ to use in the advisory communication

    > __Example:__ Applications developers are unable to deploy code in XXX foundation. Support is investigating.

    Avoid drawing conclusions of any type or stating what is assumed to be the real issue. The problem statement is for the purpose of informing the community that there is some perceived disruption of expected service levels. Information about troubleshooting should be introduced through the advisory updates and final update will typically provide or refer to a root cause analysis.
    The purpose of initial problem statement is to inform application teams that we are aware of an issue that may impact their applications, that we are actively investigating that issue, and what actions they may need to perform as a result of what is happening.
2. __Prepare a business impact statement__ to use in the advisory communication

    This is a statement of what the impact is to a) end users and/or b) application developers. These statements should be concluded in terms of when they started and when they finished (e.g. from HH:MM EST on mm/dd until HH:MM EST on mm/dd).

    > __Example 1:__ Users are not able to deploy applications in US Central region from 13:30 EST on 09/19 until TBD

    > __Example 2:__ Application latency on US Central region from 13:25 EST on 09/19 until 14:45 EST on 09/19.

1. __Prepare an actions required statement__ to use in the advisory communication

    Consider if you want applications to submit an incident so they can be included in follow up once the issue is believed to be resolved. Sometimes it may be necessary to convey known work arounds, but the usual assumption is that it is not known for the initial advisory notification.

    > __Example 1:__ None. Information only

    > __Example 2:__ Please submit a Serverless incident if you need to have the Serverless service team follow up with you after the problem is resolved

    > __Example 3:__ Application teams are asked to fail their applications over to other active region if they are experiencing a disruption of service.

1. __Create an incident__ to capture the issue in our words. This incident will be used for the advisory communications.

   ## Two ways to create a new incident

   ### Option: 1

   Select "Service Operations" as workspace and choose the Incident list that you have created and select "New" button at right hand side.

   ![Example](INC1.png)

   ### Option: 2

   Under "All" choose "Incidents" under "Self-service" and click on "New" to create an incident.

   ![Example](INC2.png)

   ![Example](INC3.png)

1. __Make sure__ the incident is assigned and in progress

[Back to Top](#serverless-minor-incident-advisory-process) | [Back to Main](README.md)

## Part 2 Send the Initial Communication

1. __View the incident__ in the `All | Incidents` view and select the 'Default view' view
    ![Example](default-view.png)
1. __Enter the location__, such as 'Wagner Place East'
1. __Enter the region__ (or regions) where the issue is occurring in the space next to the location
1. __Confirm the service__ is 'Platform', __configuration item__ is either '54514 [SERVERLESS]::DEV' or '54514 [SERVERLESS]::PROD', and __service offering__ is set to 'GCP Foundation'
1. __Enter the Actual Start__ from the Resolution Information tab
1. __Update additional comments__ from the Notes tab. This will become the ***initial description*** in the advisory communication. Make sure `Additional comments (Customer visible)` is checked before you select `Post`

    > __Note:__ Make sure the description includes when the next update will be provided.

1. __Go to the Incident Communication Plans__ tab at the bottom of the page
    > Note: You should always save the incident before you go to the Incident Communication Plan tab

1. __Select the Ford Incident Communication Plan__ in the list of available templates. If it is not there you may have to go back and save the incident
1. __Select the Communicate tab__ next to notes and then click `Compose` under `Actions`
1. __Add__ the appropriate bulkmail lists in the `To` field, leaving the pre-populated "INC_COMM_${REGION}@ford.com bulkmail(s) in place:

    Bulkmail: gcp_serverless_service_subscribers@ford.com; gcp_informational_outage_advisories@ford.com

1. __Add__ supervisors and team leads to the CC list
1. __Update__ the subject to `Serverless Incident Communication: INC######## - Medium - Initial` by adding `Serverless` to the beginning and `- Medium - Initial` to the end
1. __Click the full screen button__ to view the body of the communication in full screen mode
1. __Fill out the template__ with any required information that was not pulled from the incident
1. __Click Send__ to send the communication

___Example: Incident Fields & Selections___
![Example of Incident Fields](INC.png)

___Example: Incident Communication Plan Selection___
![Example of ICP Selection](ICP.png)


[Back to Top](#serverless-minor-incident-advisory-process) | [Back to Main](README.md)

## Part 3 Send the Update Communication

1. __Open the incident__ in the `All | Incidents` view and select the 'Default view' view
1. __Make sure__ you are viewing the incident and not in the Incident Communication Plan.
1. __Update additional comments__ from the Notes tab. This will become the ***updated description*** in the advisory communication.

    > __Note:__ Make sure `Additional comments (Customer visible)` is visible (work notes is unchecked) before you select `Post`

    > __Note:__ Make sure the description includes when the next update will be provided.

1. __Go to the Incident Communication Plans__ tab at the bottom of the page
    > Note: You should always save the incident before you go to the Incident Communication Plan tab

1. __Select the Ford Incident Communication Plan__ in the list of available templates
1. __Select the Communicate tab__ next to notes and then click `Compose` under `Actions`
1. __Add__ the appropriate bulkmail lists in the `To` field, leaving the pre-populated "INC_COMM_${REGION}@ford.com bulkmail(s) in place:

    Bulkmail: gcp_serverless_service_subscribers@ford.com; gcp_informational_outage_advisories@ford.com

1. __Add__ supervisors and team leads to the CC list
1. __Update__ the subject to `Serverless Incident Communication: INC######## - Medium - Update` by adding `Serverless` to the beginning and `- Medium - Update` to the end
1. __Click the full screen button__ to view the body of the communication in full screen mode
1. __Fill out the template__ with any required information that was not pulled from the incident.

    > _Hint: There should be no need for this if you followed the documented process._

1. __Click Send__ to send the communication

[Back to Top](#serverless-minor-incident-advisory-process) | [Back to Main](README.md)

## Part 4 Send the Final Communication

1. __Open the incident__ in the `All | Incidents` view and select the 'Default view' view
1. __Make sure__ you are viewing the incident and not in the Incident Communication Plan.
1. __Enter the Actual End__ from the Resolution Information tab. This would be a good time to enter the __resolution code__ and __resolution notes__ as well. 
1. __Update additional comments__ from the Notes tab. This will become the ***updated description*** in the advisory communication.

    > __Note:__ Make sure `Additional comments (Customer visible)` is visible (work notes is unchecked) before you select `Post`

    > __Note:__ Make sure the description includes when the next update will be provided.

1. __Go to the Incident Communication Plans__ tab at the bottom of the page
    > Note: You should always save the incident before you go to the Incident Communication Plan tab

1. __Select the Ford Incident Communication Plan__ in the list of available templates
1. __Select the Communicate tab__ next to notes and then click `Compose` under `Actions`
1. __Add__ the appropriate bulkmail lists in the `To` field, leaving the pre-populated "INC_COMM_${REGION}@ford.com bulkmail(s) in place:

    Bulkmail: gcp_serverless_service_subscribers@ford.com; gcp_informational_outage_advisories@ford.com

1. __Add__ supervisors and team leads to the CC list
1. __Update__ the subject to `Serverless Incident Communication: INC######## - Medium - Final` by adding `Serverless` to the beginning and `- Medium - Final` to the end
1. __Click the full screen button__ to view the body of the communication in full screen mode
1. __Fill out the template__ with any required information that was not pulled from the incident.

    > _Hint: There should be no need for this if you followed the documented process._

1. __Click Send__ to send the communication

Refer:  [Demo Video - Incident Advisory Creating Process Using ServiceNow for GCP Serverless Team
](https://videosat.ford.com/#/videos/5779e314-1eda-46c0-960a-d0d61ebafa27)

[Back to Top](#serverless-minor-incident-advisory-process) | [Back to Main](README.md)
