"""
Benchmark Runner
================
Orchestrates end-to-end ASR benchmarking: replays test audio for each language,
collects session results, and prints a summary table.

Prerequisites:
    - Server running: python server.py
    - Test audio downloaded: python scripts/download_test_audio.py

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --server localhost:8000 --languages hi ta te bn
    python scripts/benchmark.py --pause 3
"""

import argparse
import asyncio
import glob
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))
sys.path.insert(0, BASE_DIR)

from replay_client import replay_files


LANGUAGES = ["hi", "ta", "te", "bn"]
UTTERANCES_DIR = os.path.join(BASE_DIR, "test_utterances")
SESSIONS_DIR = os.path.join(BASE_DIR, "data", "sessions")


def find_wav_files(lang: str) -> list[str]:
    """Find all WAV files for a language."""
    lang_dir = os.path.join(UTTERANCES_DIR, lang)
    files = sorted(glob.glob(os.path.join(lang_dir, "*.wav")))
    return files


def get_latest_session() -> dict | None:
    """Get the most recently modified session JSON."""
    if not os.path.exists(SESSIONS_DIR):
        return None
    jsons = sorted(
        glob.glob(os.path.join(SESSIONS_DIR, "*.json")),
        key=os.path.getmtime,
    )
    if not jsons:
        return None
    with open(jsons[-1]) as f:
        return json.load(f)


def count_sessions() -> int:
    """Count session files."""
    if not os.path.exists(SESSIONS_DIR):
        return 0
    return len(glob.glob(os.path.join(SESSIONS_DIR, "*.json")))


async def run_benchmark(server: str, languages: list[str], pause: float):
    """Run benchmark for all specified languages."""
    results = {}

    for lang in languages:
        files = find_wav_files(lang)
        if not files:
            print(f"\nNo WAV files found for {lang.upper()} in {UTTERANCES_DIR}/{lang}/")
            print(f"  Run: python scripts/download_test_audio.py -l {lang}")
            continue

        print(f"\n{'='*60}")
        print(f"  Benchmarking {lang.upper()} — {len(files)} files")
        print(f"{'='*60}")

        session_count_before = count_sessions()

        await replay_files(files, server, lang)

        # Wait for session to be written
        print(f"Waiting for session to finalize...")
        await asyncio.sleep(pause)

        # Find the new session
        session = get_latest_session()
        if session and count_sessions() > session_count_before:
            results[lang] = session
            summary = session.get("summary", {})
            print(f"\n  Results for {lang.upper()}:")
            print(f"    Utterances:     {summary.get('total_utterances', 0)}")
            print(f"    Avg Latency:    {summary.get('avg_api_latency_ms', 0)}ms")
            print(f"    Avg RTF:        {summary.get('avg_rtf', 'N/A')}")
            print(f"    Avg TTFT:       {summary.get('avg_ttft_ms', 'N/A')}ms")
            print(f"    Avg WER:        {summary.get('avg_wer', 'N/A')}")
        else:
            print(f"  Warning: no new session found for {lang}")

    return results


def print_summary(results: dict):
    """Print a consolidated summary table."""
    if not results:
        print("\nNo results to summarize.")
        return

    print(f"\n{'='*72}")
    print(f"  BENCHMARK SUMMARY — Parakeet RNNT 1.1B Indic Profile")
    print(f"{'='*72}")

    # Collect all utterances across sessions
    all_utterances = []
    for lang, session in results.items():
        for u in session.get("utterances", []):
            u["_lang"] = lang
            all_utterances.append(u)

    # Per-language table
    header = f"{'Lang':<6} {'Utts':>5} {'Avg Lat':>9} {'Min Lat':>9} {'Max Lat':>9} {'Avg RTF':>9} {'Avg TTFT':>10}"
    print(f"\n  {header}")
    print(f"  {'-' * len(header)}")

    total_utts = 0
    total_latencies = []
    total_rtfs = []

    for lang in sorted(results.keys()):
        s = results[lang].get("summary", {})
        utts = s.get("total_utterances", 0)
        total_utts += utts

        lat_vals = [u["api_latency_ms"] for u in results[lang].get("utterances", []) if u.get("api_latency_ms")]
        rtf_vals = [u["rtf"] for u in results[lang].get("utterances", []) if u.get("rtf") is not None]
        total_latencies.extend(lat_vals)
        total_rtfs.extend(rtf_vals)

        avg_lat = s.get("avg_api_latency_ms", 0)
        min_lat = s.get("min_api_latency_ms", 0)
        max_lat = s.get("max_api_latency_ms", 0)
        avg_rtf = s.get("avg_rtf")
        avg_ttft = s.get("avg_ttft_ms")

        rtf_str = f"{avg_rtf:.3f}" if avg_rtf is not None else "N/A"
        ttft_str = f"{avg_ttft}ms" if avg_ttft is not None else "N/A"

        print(f"  {lang.upper():<6} {utts:>5} {avg_lat:>7}ms {min_lat:>7}ms {max_lat:>7}ms {rtf_str:>9} {ttft_str:>10}")

    # Overall
    print(f"  {'-' * len(header)}")
    overall_lat = round(sum(total_latencies) / len(total_latencies)) if total_latencies else 0
    overall_rtf = round(sum(total_rtfs) / len(total_rtfs), 3) if total_rtfs else None
    rtf_str = f"{overall_rtf:.3f}" if overall_rtf is not None else "N/A"
    print(f"  {'ALL':<6} {total_utts:>5} {overall_lat:>7}ms {'':>9} {'':>9} {rtf_str:>9}")

    # Key takeaway
    print(f"\n  Key numbers for LinkedIn comment:")
    print(f"    Total utterances: {total_utts}")
    print(f"    Avg API latency:  {overall_lat}ms")
    print(f"    Avg RTF:          {rtf_str}")
    if overall_rtf and overall_rtf < 1.0:
        print(f"    Real-time:        YES (RTF < 1.0)")
    print(f"    Languages tested: {', '.join(l.upper() for l in sorted(results.keys()))}")

    # Save summary to file
    summary_path = os.path.join(BASE_DIR, "data", "benchmark_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_utterances": total_utts,
        "avg_api_latency_ms": overall_lat,
        "avg_rtf": overall_rtf,
        "languages": {},
    }
    for lang in sorted(results.keys()):
        s = results[lang].get("summary", {})
        summary["languages"][lang] = {
            "total_utterances": s.get("total_utterances", 0),
            "avg_api_latency_ms": s.get("avg_api_latency_ms", 0),
            "min_api_latency_ms": s.get("min_api_latency_ms", 0),
            "max_api_latency_ms": s.get("max_api_latency_ms", 0),
            "avg_rtf": s.get("avg_rtf"),
            "avg_ttft_ms": s.get("avg_ttft_ms"),
            "avg_wer": s.get("avg_wer"),
            "session_id": results[lang].get("session_id"),
        }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Parakeet Indic ASR benchmark")
    parser.add_argument("--server", default="localhost:8000", help="Server address")
    parser.add_argument("--languages", "-l", nargs="+", default=LANGUAGES,
                        help="Languages to benchmark (default: hi ta te bn)")
    parser.add_argument("--pause", type=float, default=3.0,
                        help="Seconds to wait between languages for session finalization")
    args = parser.parse_args()

    print("Parakeet RNNT 1.1B Multilingual — Indic Profile Benchmark")
    print(f"Server: {args.server}")
    print(f"Languages: {', '.join(args.languages)}")

    results = asyncio.run(run_benchmark(args.server, args.languages, args.pause))
    print_summary(results)


if __name__ == "__main__":
    main()
