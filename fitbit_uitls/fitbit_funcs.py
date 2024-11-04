import os
import json
import pandas as pd
import datetime



HEART_URL = f'curl -X GET "https://api.fitbit.com/1/user/-/activities/heart/date/today/1w.json" -H "accept: application/json" -H "authorization: Bearer '

ECG_URL = f'curl -X GET "https://api.fitbit.com/1/user/-/ecg/list.json?afterDate=2024-03-24&sort=asc&limit=1&offset=0" -H "accept: application/json" -H "authorization: Bearer '


def get_last_min_heart_data(access_token, date, lat, lon):
    hmdata = []
    final_url = HEART_URL + access_token + '"'
    os.system(f'{final_url} > lastmin.json')
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    with open("lastmin.json") as f:
        data = json.load(f)
    for d in data['activities-heart']:
        if d['dateTime'] == date:
            for elm in d.get('value').get('heartRateZones'):
                elm["date"] = date
                elm["time"] = current_time
                elm["lat"] = lat
                elm["lon"] = lon
                hmdata.append(elm)
    return pd.DataFrame(hmdata)


def get_live_heart_rate(access_token, date):
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M")
    prev_time = now - datetime.timedelta(minutes=3)
    prev_time = prev_time.strftime("%H:%M")
    heart_rate_url = f'curl -X GET "https://api.fitbit.com/1/user/-/activities/heart/date/{date}/{date}/1min/time/{prev_time}/{current_time}.json" -H "accept: application/json" -H "authorization: Bearer '
    final_url = heart_rate_url + access_token + '"'
    print(final_url)
    os.system(f'{final_url} > hrate.json')
    with open("hrate.json", "r") as f:
        data = json.load(f)
    return pd.DataFrame(data.get("activities-heart-intraday").get("dataset"))

