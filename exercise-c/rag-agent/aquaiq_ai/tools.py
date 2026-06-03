import requests
import csv
import io
from typing import Dict, Any, List

# FIPS codes: country:state:county
# as per the API

COUNTY_CODES = {
    "travis county texas": "US:48:453",
    "williamson county texas": "US:48:491",
    "benton county arkansas": "US:05:007",
    "baxter county arkansas": "US:05:005",
    "prince george county maryland": "US:24:033",
    "oklahoma county oklahoma": "US:40:109",
    "harris county texas": "US:48:201",
    "dallas county texas": "US:48:113",
}

WATER_QUALITY_TOOL = {
    "type": "function",
    "function": {
        "name": "get_water_quality",
        "description": "Get water quality monitoring sites for a county in Texas or surrounding states.",
        "parameters": {
            "type": "object",
            "properties": {
                "county_name": {
                    "type": "string",
                    "description": "County name and state, e.g., 'Travis County Texas'"
                },
                "characteristic": {
                    "type": "string",
                    "description": "Optional: 'Nitrogen', 'pH', 'Dissolved oxygen'",
                    "default": ""
                }

            },
            "required": ["county_name"]
        }
    }
}

def find_county_code(county_name: str) -> str:
    # Converting county name to FIPS code.
    name = county_name.lower().strip()
    if name in COUNTY_CODES:
        return COUNTY_CODES[name]

    for key, code in COUNTY_CODES.items():
        if key in name or name in key:
            return code
    return None

def execute_water_quality_tool(county_name: str, characteristic: str = "") -> Dict[str, Any]:
    # Fetching water quality monitoring sites from USGS Water Quality Portal.
    fips = find_county_code(county_name)
    if not fips:
        return {
            "error": f"County '{county_name}' not recognized",
            "supported_counties": list(COUNTY_CODES.keys())
        }

    # Build request - CSV format works reliably
    url = "https://www.waterqualitydata.us/data/Station/search"
    params = {
        "countycode": fips,
        "mimeType": "csv",
        "zip": "no",
    }

    if characteristic:
        params["characteristicName"] = characteristic
    try:
        print(f"  Fetching water quality data for {county_name}...")
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 204:
            return {
                "location": county_name,
                "message": f"No monitoring sites found for {county_name}",
                "sites": []
            }

        response.raise_for_status()
        # Parse CSV response
        csv_reader = csv.DictReader(io.StringIO(response.text))
        sites = []
        for i, row in enumerate(csv_reader):
            if i >= 10:  # Limit to 10 sites
                break
            site = {
                "name": row.get("MonitoringLocationName", "Unknown"),
                "type": row.get("MonitoringLocationTypeName", "Unknown"),
                "latitude": row.get("LatitudeMeasure", "N/A"),
                "longitude": row.get("LongitudeMeasure", "N/A"),
                "organization": row.get("OrganizationFormalName", "Unknown"),
            }
            sites.append(site)
        if not sites:
            return {
                "location": county_name,
                "message": f"No monitoring sites found for {county_name}",
                "sites": []
            }
        return {
            "location": county_name,
            "fips_code": fips,
            "total_sites": len(sites),
            "sites": sites,
            "message": f"Found {len(sites)} water quality monitoring sites in {county_name}"
        }
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again."}
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}

# Testing purpose
if __name__ == "__main__":
    # Test with Travis County
    result = execute_water_quality_tool("Travis County Texas")
    print(result)
