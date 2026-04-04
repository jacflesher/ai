## Change Control Process for Ford Serverless Product Stack 

Before you start the process of making a change for any of our products, please consider all of the following items before proposing the change.

1.	Has the change you are making been tested thoroughly in one of our pre-prod or sandbox environments?
2.	Has this change been discussed with the team via one of the following methods?
- Daily Stand-Up Walk-in
- Conversation via 'Cloud Platform Development - General' WebEx Teams space
- Conversation via 'Cloud Platform Development - Cloud Run | Cloud Function | Eventarc' WebEx Teams space
3.	Has there been a "rollback" plan defined if we need to revert this change due to unexpected behavior?
4.	Has the code been checked into GitHub via a Pull Request, Peer Reviewed / Tested (by somebody other than the author)> This includes but is not limited to Terraform module changes, service definitions, or Ford Cloud Portal related files 

For more info on proper process for submitting PRs, Testing, and merging changes into a 'master' branch please contact one of the team leaders

If any of these items listed above have not been met, we probably aren't quite ready to make the jump to Production just yet. 

PLEASE NOTE: If our team runs into a breaking change scenario and causes an outage in any of our environments: you must contact one of the team leads immediately to inform them of the change, what happened, and the impact. 

### Planning Your Change (Before)
1.	Is there any possibility of customer impact from this change? Will other platforms be affected by this change? Will anyone notice your change?
- If Yes, we need to submit an Information Advisory alerting customers of this change and plan a change window. Details on how to submit an Advisory can be found below.
- If No, move on to the next step
2.	If someone other than yourself has asked you to implement the change, educate yourself to ensure that you understand the change
- Verify that you can make the change and if you do, make sure you have access to revert the change if necessary
-	Test (or Retest) in a Sandbox environment before you deploy the change; especially if it's not one that you created yourself 

### Implementing the Change (During)
1.	Call out the change is being made in 'CaaS - Deployments' WebEx Teams channel
2.	Execute your change
-	In the event of unexpected behavior, send a message to the 'CaaS - Deployments' WebEx Teams channel stating that the change is exhibiting unexpected behavior. Prepare to execute the "rollback" plan for your change after engaging other team members to assess
Once the Change is Implemented (After)
1.	Run any necessary smoke-test to validate the change was successful
2.	Collect any necessary evidence to provide for the associated Rally Story and upload
3.	Call out the change has been successfully completed in 'CaaS - Deployments' WebEx
4.	If an Information Advisory was sent: Send an updated advisory stating that the change has completed successfully, steps can be found below for sending an advisory along with sample verbiage


### Post Change (After)
1. Verify that the change has been successfully implemented.
2. Document all of your change modifications in the form of a Pull Request, Issue, or CHG BMC ticket
3. Inform the relevent WebEx channel of the change information and that it was successful 'Cloud Platform Development - Cloud Run | Cloud Function | Eventarc' 
4. Close the change with full notes including how to roll back and who to contact with further questions 

# Informational Change Advisory

Note: An informational advisory should be created for customers after these conditions have been met. The change has been fully tested by a Serverless team member, the change has been approved by the LL6, and a discussion on timing and impacted parties has taken place.

Follow the process for creating a __informational change advisory__

In ServiceNow process for informational advisories is to send emails using the standard IT advisory template (Word doc) as a guide for standard format for informational advisories, including our change notifications, going forward. 

Link to the KB process in ServiceNow: [IT Informational Advisory Template](https://www.techatford.com/esc?id=kb_article_view&sys_kb_id=b5606ef483f9de10c53b5ea08bda1ec4)

1. Draft Email Using Template: Open the email template that is mentioned through the link below. Fill in the required content with relevant information for your advisory. 

- Provide a start and end time.
- For application, Type Cloud Run, External Cloud Run, Cloud Functions, or Event Arc
- For Impacted region(s), Insert one or more of the following: NA, EU, GCN, SA, IMG or delete this section if it is Global
- For location,  Insert location if smaller than a region, Global, or delete this section if it’s region specific (above)
- Put in Change Number(s), Insert change number associated with the advisory or delete this section if there is no change number
- Provide a summary of the change under the “Summary” section
- In the "Situation/Background" section write detailed description of the reason causing the change, what the change will be, who the affected customers will be, and links to relevant information.
- In "Business Impact" explain the impact of the change to the customers, and its impact on the service.
- In "Action Required by the User" explain which projects/configurations may see changes if they do and do not action the direction taken in the advisory. If no action is required by the user mention that also.
- For Contact, Insert the CDSID of a person the customer can contact with any questions or problems

2. Copy Template to Email: Open Outlook and create a new email. Copy the entire template table from the Word document and paste it into the body of the email. 

3. Add Recipient List: Add your affected customer list in the ‘To:’ line of your email. If there are over 500 customers for your advisory, you should utilize a bulk mail list. You can utilize the following regional bulkmail for our advisory

gcp_informational_outage_advisories@ford.com 

4. Add Email Subject: Type Informational Advisory - Your Application Name (Your Application Number) into the ‘Subject:’ line of your email. 

5. Send Email: Double check your ‘To:’ and ‘Subject:’ lines and proof read the body of your email from any grammatical errors. Click the ‘Send’ button. If you utilized a bulk mail list for your advisory, you may get a proof email for review. Review the proof sent to your email and click ‘Certify’ when you are ready to send. 

![Sample-informational-advisory](Sample-informational-advisory.png)

