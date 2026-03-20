"""
Post-hoc Session Analyzer
==========================
Analyzes completed session JSONs from real Plivo calls against manifest
reference texts. Computes WER and summarizes TTFS + API latency.

Use this after making real phone calls where the caller reads scripted
sentences from the manifest in order.

Usage:
    python scripts/analyze_session.py data/sessions/<id>.json --language hi
    python scripts/analyze_session.py data/sessions/*.json --language hi ta bn
"""

import argparse
import json
import os
import re
import sys
import unicodedata

import jiwer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(BASE_DIR, "test_utterances", "manifest.json")


def normalize_indic(text: str) -> str:
    """Normalize text for Indic WER comparison."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[।॥,.\-!?:;"\'\(\)\[\]…\u0964\u0965]', "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def compute_wer(reference: str, hypothesis: str) -> float:
    ref = normalize_indic(reference)
    hyp = normalize_indic(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    if not hyp:
        return 1.0
    word_wer = jiwer.wer(ref, hyp)
    ref_chars = ref.replace(" ", "")
    hyp_chars = hyp.replace(" ", "")
    cer = jiwer.cer(ref_chars, hyp_chars)
    return round(min(word_wer, cer), 4)


def reassemble_utterances(utterances: list[dict], references: list[str]) -> list[dict]:
    """Align session utterance fragments to reference sentences.

    Strategy: concatenate all utterance texts, then greedily split by
    best WER alignment against each reference sentence in order.

    For small numbers of utterances (2-3 calls), a simpler approach:
    group consecutive utterance fragments and match against references
    sequentially.
    """
    if not utterances or not references:
        return []

    # Simple sequential grouping: assign fragments to references
    # based on fragment count roughly proportional to reference length
    all_texts = [u["text"] for u in utterances]
    all_latencies = [u["api_latency_ms"] for u in utterances]

    # Try to split fragments evenly across references
    n_refs = len(references)
    n_frags = len(all_texts)

    if n_frags <= n_refs:
        # One fragment per reference (or fewer fragments than refs)
        groups = [[t] for t in all_texts]
        latency_groups = [[l] for l in all_latencies]
        # Pad with empty groups
        while len(groups) < n_refs:
            groups.append([])
            latency_groups.append([])
    else:
        # More fragments than references — distribute proportionally
        frags_per_ref = n_frags / n_refs
        groups = []
        latency_groups = []
        start = 0
        for i in range(n_refs):
            end = round(frags_per_ref * (i + 1))
            groups.append(all_texts[start:end])
            latency_groups.append(all_latencies[start:end])
            start = end

    results = []
    for i, ref in enumerate(references):
        if i >= len(groups):
            break
        hypothesis = " ".join(groups[i]) if groups[i] else ""
        avg_latency = (
            round(sum(latency_groups[i]) / len(latency_groups[i]))
            if latency_groups[i] else None
        )
        wer = compute_wer(ref, hypothesis) if hypothesis else 1.0
        results.append({
            "reference": ref,
            "hypothesis": hypothesis,
            "wer": wer,
            "api_latency_ms": avg_latency,
            "fragment_count": len(groups[i]),
        })

    return results


def analyze_session(session_path: str, language: str, references: list[str]):
    """Analyze a single session file."""
    with open(session_path) as f:
        session = json.load(f)

    utterances = session.get("utterances", [])
    if not utterances:
        print(f"  No utterances in {os.path.basename(session_path)}")
        return None

    print(f"\n  Session: {session['session_id']}")
    print(f"  File:    {os.path.basename(session_path)}")
    print(f"  Started: {session['started_at']}")
    print(f"  Utterances: {len(utterances)}")

    aligned = reassemble_utterances(utterances, references)

    for i, r in enumerate(aligned, 1):
        wer_str = f"{r['wer']:.0%}" if r["hypothesis"] else "MISS"
        lat_str = f"{r['api_latency_ms']}ms" if r["api_latency_ms"] else "N/A"
        print(f"\n    [{i}] WER={wer_str}  API={lat_str}  frags={r['fragment_count']}")
        print(f"        ref: {r['reference'][:70]}")
        if r["hypothesis"]:
            print(f"        hyp: {r['hypothesis'][:70]}")

    # Summary
    wers = [r["wer"] for r in aligned if r["hypothesis"]]
    latencies = [r["api_latency_ms"] for r in aligned if r["api_latency_ms"]]

    avg_wer = sum(wers) / len(wers) if wers else None
    avg_lat = round(sum(latencies) / len(latencies)) if latencies else None

    # TTFS from session: use finalization_ms if available
    ttfs_vals = [u["finalization_ms"] for u in utterances if u.get("finalization_ms")]
    avg_ttfs = round(sum(ttfs_vals) / len(ttfs_vals)) if ttfs_vals else None

    summary = {
        "session_id": session["session_id"],
        "language": language,
        "utterances": len(aligned),
        "avg_wer": round(avg_wer, 4) if avg_wer is not None else None,
        "avg_api_latency_ms": avg_lat,
        "avg_ttfs_ms": avg_ttfs,
        "aligned": aligned,
    }

    print(f"\n    Summary: WER={avg_wer:.1%}  API={avg_lat}ms  TTFS={avg_ttfs}ms"
          if avg_wer is not None else "\n    Summary: insufficient data")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze session JSONs against manifest")
    parser.add_argument("sessions", nargs="+", help="Session JSON file(s)")
    parser.add_argument("--language", "-l", nargs="+", required=True,
                        help="Language(s) of the sessions (in order)")
    parser.add_argument("--manifest", default=MANIFEST_PATH, help="Manifest JSON path")
    args = parser.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))

    print("Post-hoc Session Analysis")
    print(f"Manifest: {args.manifest}")

    for i, session_path in enumerate(args.sessions):
        lang = args.language[i] if i < len(args.language) else args.language[0]
        references = [e["reference_text"] for e in manifest if e["lang"] == lang]

        if not references:
            print(f"\n  No references for language '{lang}' in manifest")
            continue

        analyze_session(session_path, lang, references)


if __name__ == "__main__":
    main()
