import requests

def get_air_quality(city: str):
   try:
       geo_url = "https://geocoding-api.open-meteo.com/v1/search"
       geo_res = requests.get(geo_url, params={"name": city}, timeout=5)
       geo_res.raise_for_status()
       geo_data = geo_res.json()
       if "results" not in geo_data:
           return f"City not found: {city}"
       lat = geo_data["results"][0]["latitude"]
       lon = geo_data["results"][0]["longitude"]
       aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
       params = {
           "latitude": lat,
           "longitude": lon,
           "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone"
       }
       aq_res = requests.get(aq_url, params=params, timeout=5)
       aq_res.raise_for_status()
       aq_data = aq_res.json()
       i = -1
       return f"""
Air quality in {city}:
PM2.5: {aq_data['hourly']['pm2_5'][i]}
PM10: {aq_data['hourly']['pm10'][i]}
NO2: {aq_data['hourly']['nitrogen_dioxide'][i]}
Ozone: {aq_data['hourly']['ozone'][i]}
"""
   except Exception as e:
       return f"Error fetching air quality: {str(e)}"