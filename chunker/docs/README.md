# GraphQL Secret Rotation Process

## Prerequisites

Before beginning, ensure you have:

- **MFA access** to both `sservic7@ford.com` and `ebot1@ford.com`
- **Git** and **gcloud CLI** installed on your local machine

---

## Initial Setup

### 1. Clone the Repository

```sh
git clone https://github.com/ford-cloud/serverless-scripts && cd serverless-scripts
```

### 2. Authenticate with gcloud

```sh
gcloud auth login
```

---

## Rotation Process

### Option A: Using Helper Scripts (Recommended for Experienced Users)

For those already familiar with the manual process, use these helper scripts:

- **ebot1 account**: [ebot1 refresh helper script](https://github.com/ford-cloud/serverless-scripts/blob/main/generate_auth_code_url_e1.sh)
- **sservic7 account**: [sservic7 refresh helper script](https://github.com/ford-cloud/serverless-scripts/blob/main/generate_auth_code_url_s7.sh)

If using the helper scripts, you may skip directly to [Testing](#testing).

---

### Option B: Manual Process (Recommended for First-Time Users)

It's recommended to complete the manual process at least once to understand each step. This knowledge will help troubleshoot if the helper scripts fail.

#### Step 1: Set Environment Variables

Copy and paste this snippet into your terminal:

```sh
GRAPHQL_CLIENT="$(gcloud secrets versions access "latest" \
 --project "ford-4dbd7038019cc45cc32a5084" \
 --secret "GRAPHQL_CLIENT")"
GRAPHQL_SECRET="$(gcloud secrets versions access "latest" \
 --project "ford-4dbd7038019cc45cc32a5084" \
 --secret "GRAPHQL_SECRET")"
STATE=$(hexdump -n16 -e '1/1 "%02x"' /dev/urandom)
printf '\nGRAPHQL_CLIENT: %s\nGRAPHQL_SECRET: %s\nSTATE: %s\n\n' \
    "$GRAPHQL_CLIENT" "$GRAPHQL_SECRET" "$STATE"
```

> **⚠️ SECURITY WARNING**  
> The `GRAPHQL_SECRET` is extremely sensitive and must be rotated annually. **Never store it locally.** Always retrieve a fresh secret in case it has changed. The `GRAPHQL_CLIENT` will never change and is not sensitive, so it may be stored locally if desired.

#### Step 2: Generate the Authorization URL

Run this command to build the authorization URL:

```sh
printf '\n%s\n\n' "https://login.microsoftonline.com/azureford.onmicrosoft.com/oauth2/v2.0/authorize?client_id=${GRAPHQL_CLIENT}&response_type=code&redirect_uri=https://teamsintegrationtest&response_mode=query&scope=offline_access%20User.Read%20ChannelMessage.Send%20Chat.ReadWrite&state=${STATE}"
```

Copy the URL from the output.

#### Step 3: Authenticate via Browser

1. Open **Firefox Private Browser**
2. Navigate to the URL you just copied
3. When prompted to log in, authenticate with the account you are updating:
   - `sservic7@ford.com` **OR**
   - `ebot1@ford.com`

> **⚠️ CRITICAL**  
> Make sure you log in with the **correct service account**, not your personal account. This is a common mistake.

#### Step 4: Extract the Authorization Code

After logging in, the browser will display what appears to be an error page. However, check the **address bar**—it should contain a URL similar to:

```
https://teamsintegrationtest/?code=1.ARIAeruQyfRRm0O9NpwH-xBBwDpvs6dNBMlHu-lUvlVeFX8SAOESAA&state=aeef780f-6c8f-4b15-b1ae-663c4d8e3dc4&session_state=007c15b9-c5b4-5d1a-b157-60248baf889d
```

If you see this, everything has worked correctly up to this point.

> **⏱️ TIME-SENSITIVE**  
> You have **5 minutes** to complete the next steps before the code expires!

**Extract the code value:**

The code is everything between `?code=` and the first `&`. In the example above:

```
1.ARIAeruQyfRRm0O9NpwH-xBBwDpvs6dNBMlHu-lUvlVeFX8SAOESAA
```

> **Note:** The example code shown here has been shortened for security purposes. The real code will be **much longer**.

#### Step 5: Set the CODE Environment Variable

**Option 1 - Manual:**

```sh
export CODE="1.ARIAeruQyfRRm0O9NpwH-xBBwDpvs6dNBMlHu-lUvlVeFX8SAOESAA"
```

**Option 2 - Automated parsing:**

```sh
# Update the URL here to match the one from your browser
URL="https://teamsintegrationtest/?code=1.ARIAeruQyfRRm0O9NpwH-xBBwDpvs6dNBMlHu-lUvlVeFX8SAOESAA&state=aeef780f-6c8f-4b15-b1ae-663c4d8e3dc4&session_state=007c15b9-c5b4-5d1a-b157-60248baf889d"

# Parse out the code
CODE="$(printf '%s' "$URL" | awk -F'=' '{print $2}' | awk -F'&' '{print $1}')"

# Validate code is set properly
printf '\nCODE: %s\n\n' "$CODE"
```

#### Step 6: Obtain the Refresh Token

Execute this curl command:

```sh
curl "https://login.microsoftonline.com/azureford.onmicrosoft.com/oauth2/v2.0/token" \
--location --silent --request "POST" \
--proxy "internet.ford.com:83" \
--header "Content-Type: application/x-www-form-urlencoded" \
--data-urlencode "client_id=$GRAPHQL_CLIENT" \
--data-urlencode "client_secret=$GRAPHQL_SECRET" \
--data-urlencode "code=$CODE" \
--data-urlencode "redirect_uri=https://teamsintegrationtest" \
--data-urlencode "grant_type=authorization_code"
```

**Understanding the Response:**

- **Success**: You'll receive JSON output containing `access_token` and `refresh_token`. Extract and save the **refresh_token** value—you'll need it in the next section.

- **Failure**: If you receive the error "*The provided authorization code or refresh token has expired due to inactivity*", the code has expired. You must restart the entire process from Step 1 with a new `STATE` value.

> **Note:** After this curl command executes, both the `STATE` and `CODE` values expire immediately. They are single-use only with a 5-minute TTL.

---

## Update Secret Manager

Complete the following steps for **all three projects**:

- `ford-afd20ec4c8a9b3a7599c2ef8`
- `ford-4dbd7038019cc45cc32a5084`
- `ford-bf4f3f52c6c80b800ea99472`

### 1. Identify the Correct Secret

Depending on which account you're refreshing:

| Account | Secret Name |
|---------|-------------|
| `ebot1@ford.com` | `GRAPHQL_EIASBOT_TOKEN` |
| `sservic7@ford.com` | `GRAPHQL_TOKEN` |

### 2. Document Current Access Permissions

Before making changes, record which service accounts currently have access. Run this command to display all current access grants:

```sh
for SECRET_NAME in GRAPHQL_EIASBOT_TOKEN GRAPHQL_TOKEN; do
    for PROJECT in "ford-afd20ec4c8a9b3a7599c2ef8" "ford-4dbd7038019cc45cc32a5084" "ford-bf4f3f52c6c80b800ea99472"; do
        printf '%s - %s\n' "$PROJECT" "$SECRET_NAME"
        gcloud secrets get-iam-policy "$SECRET_NAME" \
         --project "$PROJECT" \
         --format "value(bindings.members)"
        printf '%s\n' "------------------------------"
    done
done
```

Save this output—you'll need to restore these permissions later.

### 3. Delete the Existing Secret

In the GCP Console, navigate to **Secret Manager** for each project and delete the appropriate secret.

> **Why delete?** Deleting ensures all services use the same version, avoiding issues with pinned versions containing expired secrets.

### 4. Recreate the Secret

1. In **Secret Manager**, create a new secret with the **same name** as before
2. Use the **refresh_token** value obtained from the curl command in Step 6

### 5. Restore Access Permissions

For each service account that previously had access (from Step 2):

1. Navigate to the secret's page in GCP Console
2. Click the **Permissions** tab
3. Click **Grant Access**
4. Add the service account with the role **Secret Manager Secret Accessor**

Repeat for all three projects.

---

## Testing

Verify the rotation was successful by sending a test message to yourself in MS Teams using these scripts:

**For ebot1:**
```sh
./send_message_as_ebot1.sh "$(whoami)@ford.com" "Test message"
```

**For sservic7:**
```sh
./send_message_as_sservic7.sh "$(whoami)@ford.com" "Test message"
```

**Script Locations:**
- [ebot1 send message script](https://github.com/ford-cloud/serverless-scripts/blob/main/send_message_as_ebot1.sh)
- [sservic7 send message script](https://github.com/ford-cloud/serverless-scripts/blob/main/send_message_as_sservic7.sh)

If you receive the test message in MS Teams, the rotation was successful!

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Code expired error | Restart from Step 1; you have only 5 minutes to complete Steps 4-6 |
| Wrong account logged in | Clear browser session and repeat Step 3 with correct service account |
| Test message not received | Verify secret was created correctly and permissions were restored in all three projects |