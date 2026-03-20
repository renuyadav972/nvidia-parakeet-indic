"""
Make Outbound Plivo Call
========================
Initiates an outbound call via Plivo REST API pointing to the
parakeet server's answer_url (ngrok tunnel).

The caller reads scripted sentences from the manifest, and the
server transcribes + records the call.

Usage:
    python scripts/make_call.py --to +91XXXXXXXXXX --ngrok https://xxxx.ngrok-free.app

    # With language override (default: hi)
    python scripts/make_call.py --to +91XXXXXXXXXX --ngrok https://xxxx.ngrok-free.app --language ta
"""

import argparse
import json
import os
import sys

import httpx
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(BASE_DIR, "test_utterances", "manifest.json")

# Load env from both project and turn-detection-demo (for Plivo creds)
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(os.path.join(BASE_DIR, "..", "turn-detection-demo", ".env"))


def get_plivo_numbers():
    """Try to find Plivo phone numbers from env."""
    auth_id = os.getenv("PLIVO_AUTH_ID", "")
    auth_token = os.getenv("PLIVO_AUTH_TOKEN", "")
    if not auth_id or not auth_token:
        return None

    resp = httpx.get(
        f"https://api.plivo.com/v1/Account/{auth_id}/Number/",
        auth=(auth_id, auth_token),
        params={"limit": 5},
    )
    if resp.status_code == 200:
        numbers = resp.json().get("objects", [])
        return [n["number"] for n in numbers]
    return None


def make_call(from_number: str, to_number: str, answer_url: str,
              auth_id: str, auth_token: str):
    """Initiate an outbound Plivo call."""
    resp = httpx.post(
        f"https://api.plivo.com/v1/Account/{auth_id}/Call/",
        auth=(auth_id, auth_token),
        json={
            "from": from_number,
            "to": to_number,
            "answer_url": answer_url,
            "answer_method": "POST",
        },
    )
    return resp.status_code, resp.json()


def print_script(language: str):
    """Print the sentences the caller should read."""
    manifest = json.load(open(MANIFEST_PATH, encoding="utf-8"))
    entries = [e for e in manifest if e["lang"] == language]

    print(f"\n{'=' * 60}")
    print(f"  READ THESE SENTENCES ({language.upper()}) — clearly, with pauses")
    print(f"{'=' * 60}")
    for i, entry in enumerate(entries, 1):
        print(f"\n  {i}. {entry['reference_text']}")
    print(f"\n  (pause 3-4 seconds between each sentence)")
    print(f"  (hang up when done)")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Make outbound Plivo call for benchmark")
    parser.add_argument("--to", required=True, help="Phone number to call (e.g., +91XXXXXXXXXX)")
    parser.add_argument("--from-number", help="Plivo number to call from (auto-detected if not set)")
    parser.add_argument("--ngrok", required=True, help="Ngrok HTTPS URL (e.g., https://xxxx.ngrok-free.app)")
    parser.add_argument("--language", "-l", default="hi", help="Language for the script (hi, ta, bn)")
    args = parser.parse_args()

    auth_id = os.getenv("PLIVO_AUTH_ID", "")
    auth_token = os.getenv("PLIVO_AUTH_TOKEN", "")

    if not auth_id or not auth_token:
        print("ERROR: PLIVO_AUTH_ID and PLIVO_AUTH_TOKEN not found in .env")
        print("Copy them from turn-detection-demo/.env or set them directly:")
        print(f"  echo 'PLIVO_AUTH_ID=xxx' >> {BASE_DIR}/.env")
        print(f"  echo 'PLIVO_AUTH_TOKEN=xxx' >> {BASE_DIR}/.env")
        sys.exit(1)

    # Find a from number
    from_number = args.from_number
    if not from_number:
        print("Looking up Plivo phone numbers...")
        numbers = get_plivo_numbers()
        if numbers:
            from_number = numbers[0]
            print(f"Using Plivo number: {from_number}")
        else:
            print("ERROR: No Plivo numbers found. Use --from-number to specify one.")
            sys.exit(1)

    # Build answer URL with language
    answer_url = f"{args.ngrok}/?language={args.language}"

    # Show the script first
    print_script(args.language)

    input("Press ENTER to initiate the call...")

    print(f"\nCalling {args.to} from {from_number}...")
    print(f"Answer URL: {answer_url}")

    status, resp = make_call(from_number, args.to, answer_url, auth_id, auth_token)

    if status in (200, 201):
        call_uuid = resp.get("request_uuid", "unknown")
        print(f"\nCall initiated! UUID: {call_uuid}")
        print(f"\nNow:")
        print(f"  1. Answer your phone")
        print(f"  2. Read the sentences above clearly")
        print(f"  3. Pause 3-4 seconds between sentences")
        print(f"  4. Hang up when done")
        print(f"\nThe recording will be saved automatically to data/sessions/")
        print(f"Watch the server logs and dashboard at http://localhost:8000")
    else:
        print(f"\nCall failed! Status: {status}")
        print(f"Response: {json.dumps(resp, indent=2)}")


if __name__ == "__main__":
    main()
