"""
Generate LinkedIn Comment
=========================
Reads benchmark results and generates a LinkedIn comment draft.

Usage:
    python scripts/generate_comment.py
    python scripts/generate_comment.py --summary data/benchmark_summary.json
    python scripts/generate_comment.py --tone technical
"""

import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_summary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_sessions() -> list[dict]:
    """Load all session files for detailed metrics."""
    sessions_dir = os.path.join(BASE_DIR, "data", "sessions")
    if not os.path.exists(sessions_dir):
        return []
    sessions = []
    for fname in sorted(os.listdir(sessions_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(sessions_dir, fname)) as f:
                sessions.append(json.load(f))
    return sessions


def compute_detailed_metrics(sessions: list[dict]) -> dict:
    """Compute detailed per-language metrics from session data."""
    by_lang = {}
    for s in sessions:
        for u in s.get("utterances", []):
            lang = u.get("language", "unknown")
            if lang not in by_lang:
                by_lang[lang] = {"latencies": [], "rtfs": [], "ttfts": [], "wers": [], "count": 0}
            by_lang[lang]["count"] += 1
            if u.get("api_latency_ms"):
                by_lang[lang]["latencies"].append(u["api_latency_ms"])
            if u.get("rtf") is not None:
                by_lang[lang]["rtfs"].append(u["rtf"])
            if u.get("ttft_ms") is not None:
                by_lang[lang]["ttfts"].append(u["ttft_ms"])
            if u.get("wer") is not None:
                by_lang[lang]["wers"].append(u["wer"])

    result = {}
    for lang, d in by_lang.items():
        result[lang] = {
            "count": d["count"],
            "avg_latency_ms": round(sum(d["latencies"]) / len(d["latencies"])) if d["latencies"] else None,
            "p50_latency_ms": round(sorted(d["latencies"])[len(d["latencies"]) // 2]) if d["latencies"] else None,
            "p95_latency_ms": round(sorted(d["latencies"])[int(len(d["latencies"]) * 0.95)]) if d["latencies"] else None,
            "avg_rtf": round(sum(d["rtfs"]) / len(d["rtfs"]), 3) if d["rtfs"] else None,
            "avg_ttft_ms": round(sum(d["ttfts"]) / len(d["ttfts"])) if d["ttfts"] else None,
            "avg_wer": round(sum(d["wers"]) / len(d["wers"]), 4) if d["wers"] else None,
        }
    return result


LANG_NAMES = {
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
}


def generate_comment(summary: dict, detailed: dict | None = None, tone: str = "technical") -> str:
    """Generate a LinkedIn comment from benchmark data."""
    total_utts = summary.get("total_utterances", 0)
    avg_latency = summary.get("avg_api_latency_ms", 0)
    avg_rtf = summary.get("avg_rtf")
    langs = sorted(summary.get("languages", {}).keys())

    # Filter to languages that actually produced results
    if detailed:
        langs = [l for l in langs if detailed.get(l, {}).get("count", 0) > 0]
    lang_names = [LANG_NAMES.get(l, l.upper()) for l in langs]

    # Recompute aggregate from only working languages
    if detailed:
        all_lats = []
        all_rtfs = []
        working_utts = 0
        for l in langs:
            d = detailed[l]
            if d.get("avg_latency_ms") is not None:
                all_lats.extend([d["avg_latency_ms"]] * d["count"])
            if d.get("avg_rtf") is not None:
                all_rtfs.extend([d["avg_rtf"]] * d["count"])
            working_utts += d["count"]
        if working_utts > 0:
            total_utts = working_utts
        if all_lats:
            avg_latency = round(sum(all_lats) / len(all_lats))
        if all_rtfs:
            avg_rtf = round(sum(all_rtfs) / len(all_rtfs), 3)

    # Compute headline multiplier
    rtf_multiplier = f"{1/avg_rtf:.0f}x" if avg_rtf and avg_rtf > 0 else ""

    lines = []
    lines.append(f"We put Parakeet RNNT 1.1B (Indic profile) through a real telephony pipeline at Plivo. No clean studio audio — 8kHz mulaw over actual phone lines, resampled to 16kHz.")
    lines.append("")
    lines.append(f"Numbers across {total_utts} utterances ({', '.join(lang_names)}):")
    lines.append("")
    if avg_rtf is not None:
        lines.append(f"RTF {avg_rtf:.2f} — {rtf_multiplier} faster than real-time")
    lines.append(f"Avg API latency: {avg_latency}ms per utterance")
    lines.append("")

    if detailed:
        for lang in langs:
            d = detailed.get(lang, {})
            name = LANG_NAMES.get(lang, lang.upper())
            lat = d.get("avg_latency_ms")
            rtf = d.get("avg_rtf")
            if lat is None:
                continue
            rtf_s = f"RTF {rtf:.2f}" if rtf is not None else ""
            lines.append(f"  {name}: {lat}ms avg, {rtf_s}")

    lines.append("")
    lines.append("Built a live metrics dashboard (Chart.js + FastAPI + SSE) to watch per-utterance latency, RTF, and TTFT as calls come in. The model handles telephony-grade audio without flinching.")
    lines.append("")
    lines.append("#NIM #ASR #VoiceAI #Indic #NVIDIA")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LinkedIn comment from benchmark results")
    parser.add_argument("--summary", default=os.path.join(BASE_DIR, "data", "benchmark_summary.json"),
                        help="Path to benchmark_summary.json")
    parser.add_argument("--tone", choices=["technical", "casual"], default="technical",
                        help="Comment tone (default: technical)")
    parser.add_argument("--output", "-o", default=None, help="Write comment to file")
    args = parser.parse_args()

    if not os.path.exists(args.summary):
        print(f"Error: No benchmark summary found at {args.summary}")
        print(f"Run the benchmark first: python scripts/benchmark.py")
        sys.exit(1)

    summary = load_summary(args.summary)
    sessions = load_sessions()
    detailed = compute_detailed_metrics(sessions) if sessions else None

    comment = generate_comment(summary, detailed, args.tone)

    print("=" * 60)
    print("  LINKEDIN COMMENT DRAFT")
    print("=" * 60)
    print()
    print(comment)
    print()
    print("=" * 60)
    print(f"  Character count: {len(comment)}")
    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            f.write(comment)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
