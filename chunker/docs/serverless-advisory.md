# Serverless Incident Advisory

This document describes how to use the advisory process to communicate with Serverless customers that there is a problem impacting multiple applications. This process is not to be used for change advisories or informational advisories.

## Table of Contents
* [Initial Discovery](#initial-discovery)
* [Advisory Updates](#advisory-updates)
* [Final Update](#final-update)

[Back to Main](README.md)

# Initial Discovery

Upon noticing there may be an issue with the Serverless platform, the following steps should be performed.

1. __Assess the impact__ to production applications.

| High/Critical Impact | Medium/Low Impact |
|-------------------|----------------------|
| __Assessment Criteria:__<br>* __Critical__ means the issue could potentially impact multiple core business functions or cause significant disruptions across multiple regions. For example, the issue has the potential to affect many business customers, either directly or indirectly (i.e. disrupts manufacturing, critical business services, connected vehicle functionalities, critical financial services, or significantly affects business operations)<br>* __High__ means the issue could potentially severely impact specific manufacturing processes, critical business functions, etc; but counter measures are in place that have reduced the impact, such as automatic failover to another region or manual work-arounds for affected applications that are not considered major overhead or cause more than 20% extra work. | __Assessment Criteria:__<br>* __Medium__ means the issue does not affect production workloads (i.e. exists only in Preprod environments) or the impact to production is such that production workloads are largely unaffected (i.e. intermittent latency or issue affects error logging)<br>* __Low__ means there is no significant impact to production or major disruptions to a non-prod environment |
| __Action to Take:<br>__ Follow the [Major Incident Advisory Process](major-incident-advisory.md) | __Action to Take:<br>__ Follow the [Minor Incident Advisory Process](minor-incident-advisory.md) |


[Back to Top](#serverless-incident-advisory) | [Back to Main](README.md)



# Advisory Updates

Each advisory update should introduce some new information or high level of troubleshooting steps taken without drawing major conclusions. These updates will be provided to the GIM team either through chat or on the bridge during the event triage. The final advisory update should be the only advisory update to determine root cause.

[Back to Top](#serverless-incident-advisory) | [Back to Main](README.md)

# Final Update

The final advisory update should give some indication that a root cause has been identified and a high level description of what it is. In some cases, this may be provided at a much later time but the advisory can still be concluded with some indication that the problem management process will be used to provide root cause analysis. At the conclusion of the event and final update has been sent, the advisory incident will be returned to the Serverless service team queue

[Back to Top](#serverless-incident-advisory) | [Back to Main](README.md)
