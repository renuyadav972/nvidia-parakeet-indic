"""
Compute WER
============
Offline WER computation from session JSON files against reference transcripts
in test_utterances/manifest.json.

Usage:
    python scripts/compute_wer.py data/sessions/*.json
    python scripts/compute_wer.py data/sessions/*.json --manifest test_utterances/manifest.json
"""

import argparse
import json
import sys
import unicodedata

import jiwer


def load_manifest(path: str) -> dict:
    """Load manifest and index by language+filename."""
    with open(path) as f:
        entries = json.load(f)
    index = {}
    for entry in entries:
        key = f"{entry['lang']}:{entry['file']}"
        index[key] = entry["reference_text"]
    return index


def compute_session_wer(session_path: str, manifest: dict | None = None) -> dict:
    """Compute WER for each utterance in a session."""
    with open(session_path) as f:
        session = json.load(f)

    results = []
    for utt in session.get("utterances", []):
        ref = utt.get("reference_text", "")
        hyp = utt.get("text", "")

        if not ref or not hyp:
            continue

        ref = unicodedata.normalize("NFC", ref.strip())
        hyp = unicodedata.normalize("NFC", hyp.strip())

        wer = jiwer.wer(ref, hyp)
        results.append({
            "utterance_num": utt["utterance_num"],
            "language": utt.get("language", "?"),
            "reference": ref,
            "hypothesis": hyp,
            "wer": round(wer, 4),
        })

    avg_wer = sum(r["wer"] for r in results) / len(results) if results else None
    return {
        "session_id": session.get("session_id"),
        "utterances": results,
        "avg_wer": round(avg_wer, 4) if avg_wer is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute WER from session JSON files")
    parser.add_argument("sessions", nargs="+", help="Session JSON files")
    parser.add_argument("--manifest", default=None, help="Manifest JSON file")
    args = parser.parse_args()

    manifest = None
    if args.manifest:
        manifest = load_manifest(args.manifest)

    for path in args.sessions:
        result = compute_session_wer(path, manifest)
        print(f"\nSession: {result['session_id']}")
        print(f"  Avg WER: {result['avg_wer']}")
        for u in result["utterances"]:
            print(f"  [{u['utterance_num']}] {u['language']} WER={u['wer']:.2%}")
            print(f"    REF: {u['reference']}")
            print(f"    HYP: {u['hypothesis']}")


if __name__ == "__main__":
    main()
