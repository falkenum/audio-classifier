import requests
import json
import common

# with open(common.FREESOUND_AUTH_PATH) as f:
#     auth_info = json.load(f)

# refresh_token = auth_info["refresh_token"]
data = {
    "client_id": "6HYgRcTeJuHNKvw2Zv22",
    "client_secret": "Dqi6gI1BZTuR5pjuZnu91u0iTeEm8MhNs4YPhlWm",
    "grant_type": "authorization_code",
    # "refresh_token": refresh_token
    "code": "67NwSFp5sW9eZILo41UPlepsremux8",
}

response = requests.post("https://freesound.org/apiv2/oauth2/access_token/", data=data)
with open(common.FREESOUND_AUTH_PATH, "w") as f:
    auth_info = json.loads(response.content)
    json.dump(auth_info, f)
