import requests
import json

# Function to get access token
def get_access_token(client_id, client_secret, redirect_uri, authorization_code):
    token_url = 'https://api.fitbit.com/oauth2/token'
    data = {
        'client_id': client_id,
        'grant_type': 'authorization_code',
        'redirect_uri': redirect_uri,
        'code': authorization_code
    }
    response = requests.post(token_url, data=data, auth=(client_id, client_secret))
    return response.json()

# Function to get activity data
def get_activity_data(access_token, date):
    url = f'https://api.fitbit.com/1/user/-/activities/date/{date}.json'
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

# Example usage
if __name__ == "__main__":
    # Replace with your credentials and authorization code
    CLIENT_ID = '23RXF9'
    CLIENT_SECRET = 'ca2fedc9a4579844b266ca216b046253'
    REDIRECT_URI = 'http://127.0.0.1:5000/?code=94856b5e5acd5d62ec6860b9260c874bcbfa8d6a&state=1d191b6g5o5z30185l4l2t3s4v4v6k6m#_=_'
    AUTHORIZATION_CODE = '94856b5e5acd5d62ec6860b9260c874bcbfa8d6a'
    
    # Get access token
    token_data = get_access_token(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, AUTHORIZATION_CODE)
    access_token = token_data['{"activities-heart":[{"customHeartRateZones":[],"dateTime":"2024-05-26","heartRateZones":[{"caloriesOut":2.8485,"max":119,"min":30,"minutes":3,"name":"Out of Range"},{"caloriesOut":0,"max":141,"min":119,"minutes":0,"name":"Fat Burn"},{"caloriesOut":0,"max":170,"min":141,"minutes":0,"name":"Cardio"},{"caloriesOut":0,"max":220,"min":170,"minutes":0,"name":"Peak"}],"value":"63.5"}],"activities-heart-intraday":{"dataset":[{"time":"22:59:00","value":62},{"time":"23:00:00","value":59},{"time":"23:01:00","value":62},{"time":"23:02:00","value":71}],"datasetInterval":1,"datasetType":"minute"}}']
    
    # Fetch activity data
    date = '2024-05-01'  # Example date
    activity_data = get_activity_data(access_token, date)
    
    # Print the activity data
    print(json.dumps(activity_data, indent=4))
