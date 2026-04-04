# FCP Bundle

FCP (Ford Cloud Portal) is the automation web portal where users can request to create GCP projects. It has many bundles/tiles based on the services to create GCP projects with selected services by enabling respective APIs.

Bundle is the collection of GCP services, service accounts, Tekton pipeline, etc.

## FCP Bundle Creation Flow

![Alt text](FCP-Bundle-Creation-Flow.png)

FCP Bundle creation process has below steps.

1. As we are a support team, we have to create/update respective JSON file on Bundle directory under IaC/service-definitions repository after forking it.

2. Bundle JSON file is having many sections/fields where we need to update based on our bundle requirement. You can refer the documentation [here](https://github.ford.com/IaC/service-definitions/blob/main/files/bundle_fields.md) for the same. 

2. Once JSON file creation/updation is done, raise pull request and let FCP team review and approve it. This bundle file is the backend of FCP portal.

3. Once it is approved, bundle will be visible to Ford Cloud Portal for Customer usage.

**NOTE :**
- For testing phase, the value of **enabled** should be **false** [ **"enabled": false** ]. By making it false, it wont be visible in FCP for users. But it can be accessible for us by using below URL format.

        https://www.cloudportal.ford.com/gcp/project/<bundle_json_file_name>

- Once it is tested and ready for Production, then the value of **enabled** should be changed to **true**. [ **"enabled": true** ]


## GCP Project Creation flow via FCP Bundle

![Alt text](GCP-Creation-flow.png)

1. FCP is the Front End for the user who can request for GCP Project creation through various bundles available in portal. 

2. Once request is made in FCP, it will create new repository with project name under **gcp-project** org 

3. Auto pull request will be created on newly created repository under **gcp-project** org which will trigger FCP Tekton Pipeline.

4. FCP Tekton Pipeline will provision 
    - New GCP project
    - Scaffolded repository
    - New Openshift Namespace along with Tekton Managed Terraform pipeline. 
    
    Scaffolded Repo will be created by cloning/using bundle template repo which is in serverless org (**serverless/_service_-bundle-template**) where **_service_** is cloud_run, cloud_function, etc

5. Now user will have GCP Project, New Openshift Namespace along with Tekton Managed Terraform pipeline, Scaffolded repository. Scaffolded repository will have sample terraform code to deploy services in GCP. User can update the input and can raise a pull/merge request which will trigger Managed Tekton Pipeline to create services in GCP project.
