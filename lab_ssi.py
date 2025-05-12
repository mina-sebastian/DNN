api_key = "83a78d216033fde70fcff94b3f0fae0529267bad7b6082ed98bdc4dca6cc0568"


import hashlib
import requests
import sys
import os

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sys.exit(1)

def query_virustotal(api_key, file_hash):
    url = f"https://www.virustotal.com/api/v3/files/{file_hash}"
    headers = {
        "x-apikey": api_key
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Error querying VirusTotal: {response.status_code}")
        print(response.text)
        sys.exit(1)

    data = response.json()
    try:
        detections = data['data']['attributes']['last_analysis_stats']['malicious']
        print(f"Detected as malicious by {detections} vendor(s).")
    except KeyError:
        print("Unexpected JSON structure.")
        print(data)

file_path = "C:\\Users\\chiru\\Downloads\\SSI L8 - sample1\\malware.png"

if not os.path.isfile(file_path):
    print("Invalid file path.")
    sys.exit(1)

print("[*] Calculating SHA256...")
file_hash = calculate_sha256(file_path)
print(f"[*] File SHA256: {file_hash}")

print("[*] Querying VirusTotal...")
query_virustotal(api_key, file_hash)
