# Operating within Github Cloud (github.com)

## Prerequisites

There's a get started guide on the [ford-cloud](https://github.com/ford-cloud#%EF%B8%8F-get-started) org, which should give you what you need to start contributing to repos in ford-cloud.

## Orgs

- ford-cloud: All repos within Cloud Platform
- ford-personal: Personal Repos that do not need a pipeline to run

In github.com, there are different rules applied to different orgs that change the way we do work:

- [ford-cloud](https://github.com/ford-cloud/)
  - Requires signed commits - Follow [this guide](https://github.com/ford-cloud/ford-cloud/blob/main/docs/signed-commits.md) or use our [script](https://github.com/ford-cloud/serverless-scripts/blob/main/commit_signing_onboarding_github_com.sh) to setup.
  - Not allowed to push directly to main branch
  - Requires team to do branches instead of forks
- [ford-personal](https://github.com/ford-personal)
  - Does not require signed commits
  - Cannot run any pipelines
  - Can push directly to main branch

All of the serverless team's repositories, including test repositories that need to run pipelines, need to go to the ford-cloud organization. If you are creating forks to other team's repositories, or if you have test repositories that does not need to run a pipeline, you can put it in ford-personal organization.

## Repos

All of the serverless team's repositories in ford-cloud are prefixed with the word `serverless-` except for the terraform modules (those are still named `tfm-SERVICE`) and the bundles (those are still named `sln-BUNDLE`). For each repository, there are topics (serverless, 54514, etc...) added in order to further obtain the team's repositories. This is an [example](https://github.com/search?q=topic%3A54514+org%3Aford-cloud+topic%3Aserverless&type=repositories) showing how you can use those topics to look through our repositories.

### Creation

In order to setup a repository, you need to get signed commits as a prerequisite. Follow [this guide](https://github.com/ford-cloud/ford-cloud/blob/main/docs/signed-commits.md) or use our [script](https://github.com/ford-cloud/serverless-scripts/blob/main/commit_signing_onboarding_github_com.sh) to setup. Creating a repository is the same as in github.ford.com, but there are some different things we have to do in order to setup the repo.

#### Topics

**Only those that are admins in the Github repositories can set topics. Make the below topic changes prior to removing yourself from admin.**

Click on the gear to the far right of the repository where it says "About" and add `serverless`, `54514`, and either `production` or `non-production` topics to your repo. ![Sample github repo pointing to gear on the about section to configure topics](topics.png)

#### Settings

**Only those that are admins in the Github repositories can configure Settings. Make the below changes to repository settings prior to removing yourself from admin.**

##### Custom properties

In github.com, custom properties set up most of the rulesets you will be using in your repo. Most are self explanatory, but some of the important fields here are:

- `ent_github_apps`: This property is used to install github apps used for PAC pipelines. **If your repository was scaffolded using FCP, this should be correct by default**
- `ent_classification`: Tells if project is production or non-production and will update branch protection rules **in the future**. Currently ford-cloud org created their own property (`production`), but `ent_classification` is from the github team, so `ford-cloud` will change to use this property.
- `production`: Tells if project is production or non-production and will update branch protection rules **currently**. Plans are to discontinue this and use the `ent_classification` property in the future. ![Showing all available custom properties you can configure on your repository settings](custom-properties.png)

##### Branch protection

[Most of the rules set on main branch are set based on what is configured in the custom properties](https://github.com/ford-cloud/.github-private/blob/main/BRANCH_PROTECTION.md). In order to set who can merge, the `Restrict who can push to matching branches` rule has been added. **Add the `gcp-serverless-serviceteam-admin` team if this repository is for the entire serverless team to use. If this is a personal project, you can put yourself and/or whoever is needed.** ![Showing Restrict who can push to matching branches section in branch protection rules](restrict-push-bp.png)

##### Collaborators and teams

**Only those that are admins in the Github repositories can set topics and configure settings. Make changes to [topics](#topics) and [settings](#settings) before you remove yourself from admin if you are not in the `gcp-serverless-serviceteam-admin` team.**

By default, for all of our repositories we have `gcp-serverless-serviceteam-admin` team as `admin` role and `gcp-serverless-serviceteam` as `write` role.

If you want to give suppliers, or Employee Type M the ability to see your repo, give `ENTP-GITHUB-FORD-SUPPLIER` team `read` role. There's a current issue where suppliers cannot fork out from our org to ford-personal, but they can fork from our org back to ford-cloud.
**If this is a repository used by entire serverless team, remove yourself from admin after making changes to [topics](#topics) and [settings](#settings)**

#### PAC Pipelines

**If you are migrating your scaffolded repository, or if your repository was not scaffolded using cloudportal.ford.com**, you want to setup your PAC pipeline by creating a repository object along with the github app you configured in the `custom properties` to run your PAC pipelines. You can do this one of two ways:

##### Put repository object in Openshift's argo solution, ocp-namespace-gitops

Caas is moving towards using a solution to handle injecting repository objects into the namespace. With this, you run a cldctl command passing in your repository's url, and it auto generates a PR with your repository object in a github repo. Once that repo is merged, it uses argo to auto inject the object into your namespace, and argo continuously looks at the repository object in the github repo, compares it to what's in your namespace, and will auto inject the object if it's not there.

1. [Install/Update cldctl](https://docs.ford.com/fcp/docs/cldctl/install-upgrade-cldctl/).
2. Login to github.ford.com by running `cldctl login github --git-host github.ford.com`
3. Login to azure by running `cldctl login azure`.
4. Run the below command:

  ```bash
  namespace=""
  destinationCluster=""
  ADMIN_FIM="gcp-serverless-serviceteam-admin"
  DEVELOPER_FIM="gcp-serverless-serviceteam"
  OBSERVER_FIM="gcp-serverless-serviceteam-observer"
  cldctl scaffold template \
  -n caas-namespace \
  -s caas-namespace \
  --deploy \
  --action="update" \
  -x namespace=$namespace \
  -x admins=$ADMIN_FIM \
  -x developers=$DEVELOPER_FIM \
  -x observers=$OBSERVER_FIM \
  -x cluster=$destinationCluster \
  -x pacRepoURLs="https://github.com/ford-cloud/mmaclac3-testing"
  ```

> If you need to add more than one repository url, then format the pacRepoURLs like so: `-x pacRepoURLs="https://github.com/ford-cloud/serverless-cloud-functions-cd",pacRepoURLs="https://github.com/ford-cloud/tfm-cloud-functions"`

##### Put repository object in manually

1. Clone down your repository and go to the root of your repository.
2. [Install/Update cldctl](https://docs.ford.com/fcp/docs/cldctl/install-upgrade-cldctl/).
3. Run the following command `cldctl scaffold template -s managed-terraform -n managed-terraform-repository -b BUSINESS_CODE`. The command will output that it created a file in the `tekton/manifests` folder.
4. From the root of your repository, log in to the desired cluster and switch to your namespace.
5. Upload repository to cluster by doing `kubectl apply -f .tekton/manifests/FILENAME`

### Migration

If you need to keep the migration history, follow [this](https://github.com/ford-cloud/ford-cloud/blob/main/migrations/README.md) document. If using this method:

- PAT tokens for migrating needs to have the following permissions:
  - github.ford.com PAT - admin:org, repo, workflow​
  - github.com PAT - repo, admin:org, workflow

If you don't need to keep migration history, you can start a new repository and manually copy the code over.

- If you have a PAC pipeline using tekton notify, you need to modify the pipelinerun:
  - Add this annotation to tell tekton-notify to send github comment to github.com: `tekton-notify.iac.ford.com/git-api-url: "https://api.github.com"`
  - If using microsoft teams to notify, make sure you change the value to be a [json string](https://github.com/ford-cloud/serverless-hub/commit/1ae2eeef2366b009db018254a20b291c2e4a6d54#diff-c54193a36ecee66b221a99c724b37c4e2d7d33573d68c508712749cd171670abR23-R100) instead of a regular string. If you are using a regular string, it will mess up not only microsoft teams, but the other parts of tekton notify as well.

### Contributing

- If you or the serverless team owns the repo, create `branches`.
- If you or the serverless team does not own the repo, create `forks`.
- In order to run pipelines for PRs created from a fork, the admins of the repo needs to comment `/ok-to-test` to run the pipelines.
- For more information on these types of Gitops commands you can do, see the [PAC site](https://pipelinesascode.com/docs/guide/gitops_commands/#event-type-annotation-and-dynamic-variables).
