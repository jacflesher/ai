# Serverless Major Incident Advisory
This process is to be used when there is an issue impacting multiple applications in a production environment and it is deemed to be high or critical in nature.

To create a major incident advisory, the following steps should be performed.

1. __Create an incident__ to capture the issue in our words. This incident will be used for the advisory process and creating it will allow us to hold customer incidents for follow up during or after the event.
1. __Change the incident priority__ to High or Critical
1. __Engage GIM__ and provide them with the incident that will be used for the advisory. [Click here for GIM escalation process](../incident-management/README.md#contact_gim)
1. __Propose major incident__ from the advisory incident by going to the `Create...` dropdown in the upper right hand corner and selecting _Propose Major Incident_
1. __Prepare a problem statement__ to share with the GIM representative.
    > __Example:__ Applications developers are unable to deploy code in XXX foundation. Support is investigating.

    Avoid drawing conclusions of any type or stating what is assumed to be the real issue. The problem statement is for the purpose of informing the community that there is some perceived disruption of expected service levels. Information about troubleshooting should be introduced through the advisory updates and final update will typically provide or refer to a root cause analysis.
    The purpose of initial problem statement is to inform application teams that we are aware of an issue that may impact their applications, that we are actively investigating that issue, and what actions they may need to perform as a result of what is happening.
1. __Prepare a business impact statement__ to share with the GIM representative. This is a statement of what the impact is to a) end users and/or b) application developers. These statements should be concluded in terms of when they started and when they finished (e.g. from HH:MM EST on mm/dd until HH:MM EST on mm/dd).

    > __Example 1:__ Users are not able to deploy applications in US Central region from 13:30 EST on 09/19 until TBD

    > __Example 2:__ Application latency on US Central region from 13:25 EST on 09/19 until 14:45 EST on 09/19.

1. __Prepare an actions required statement__ to share with the GIM representative. Consider if you want applications to submit an incident so they can be included in follow up once the issue is believed to be resolved. Sometimes it may be necessary to convey known work arounds, but the usual assumption is that it is not known for the initial advisory notification.

    > __Example 1:__ None. Information only
    > __Example 2:__ Please submit a Serverless incident if you need to have the Serverless service team follow up with you after the problem is resolved
    

1. __Ask GIM to send updates__ to the approprate bulkmail list as follows:

    Bulkmail: GCP_INFORMATIONAL_OUTAGE_ADVISORIES@bulkmail.ford.com                                        

[Back to Top](#serverless-major-incident-advisory) | [Back to Main](README.md)
