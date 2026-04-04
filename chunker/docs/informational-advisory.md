# Serverless - Informational Advisory

This document describes how to use the advisory process to communicate with Serverless application customers when there is information to share that does not meet the definition of incident or change advisories. This process is not to be used for incident advisories.

## Table of Contents

* [Create the Advisory](#create-the-advisory)

# Create the Advisory

Draft Email Using Template:__ Open the email template ([advisory/advisory_template](IT-Informational-Advisory-Template.docx)) Fill in the required content with relevant information for your advisory.

Link to the KB process in ServiceNow: [IT Informational Advisory Template](https://www.techatford.com/esc?id=kb_article_view&sys_kb_id=56bd24698324a6508ede8a6f5bda1ef7)

[Download RAW outlook advisory template for MAC](../advisory-process/Info-advisory-serverless-template.emltpl)

  > __Note:__  Do not change anything in the header or footer of the template. Delete section(s) from the template that not applicable to your advisory.

1. __Copy Template to Email:__ Open Outlook and create a new email. Copy the entire template table from the Word document and paste it into the body of the email.
1. __Add Recipient List:__ Add your affected customer list in the ‘To:’ line of your email:

Bulkmail: gcp_informational_outage_advisories@ford.com

3. __Add Email Subject:__ Type `Informational Advisory - Serverless(CloudRun or CloudFunction) - Brief Event Statement` into the ‘Subject:’ line of your email.
4. Fill in specific values:

  |Field           |Value                                                                                                |
  |----------------|-----------------------------------------------------------------------------------------------------|
  | Start Time     | When the event or change starts                                                                     |
  | End Time     | n/a or when the event or change will end                                                                     |
  | Application     | 54514-Serverless                                                           |
  | Impacted Region(s) | Enter the region (e.g. Global or NA, EU, GCN, SA, IMG)                                                           |
  | Location        | Enter the location (e.g. Global, GCP Central or GCP East)                                                            |
  | Change Number | Enter the change number if applicable                                                         |
  | Summary | Enter a brief event or change statement, similar to what was in the subject <br>__Example:__ _Serverless - Google support notified that they are investigating a potential issue in Cloud Run_ |
  |Affected Regions|[Check all that apply]|
  | Timing | Enter the timing as it applies to the region where the event is happening |
  | Situation Background | Enter a more detailed description of the event or change that is happening <br>__Example:__ _Serverless team is closly monitoring the impacted region and google got the situation under control and investigating further with there engineering team .|
  | Business Impact | Insert how the change or event will impact the application team customer <br>__Example:__ _Customers would observe increased latency and errors_ |
  | Action required by user | Insert the action the customer needs to make regarding the change, planned outage, or awareness <br>__Example:__ _No action required by customer. Informational Only._ |

5. __Send Email:__ Double check your ‘To:’ and ‘Subject:’ lines and proof read the body of your email from any grammatical errors. Click the ‘Send’ button. If you utilized a bulk mail list for your advisory, you may get a proof email for review. Review the proof sent to your email and click ‘Certify’ when you are ready to send. 

[Back to Top](#informational-advisory) | [Back to Main](README.md)
