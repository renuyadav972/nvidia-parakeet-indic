"""
Download Test Audio
===================
Downloads sample Indic speech utterances from Mozilla Common Voice (via HuggingFace)
for benchmarking the Parakeet Indic ASR model.

Usage:
    python scripts/download_test_audio.py
    python scripts/download_test_audio.py --samples 10
    python scripts/download_test_audio.py --languages hi ta
"""

import argparse
import json
import os
import sys
import wave

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

# Map of language code -> Common Voice dataset config name
LANG_CONFIGS = {
    "hi": "hi",   # Hindi
    "ta": "ta",   # Tamil
    "te": "te",   # Telugu
    "bn": "bn",   # Bengali
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTTERANCES_DIR = os.path.join(BASE_DIR, "test_utterances")


def convert_to_wav16k(audio_array, sampling_rate: int, output_path: str):
    """Convert audio array to 16kHz mono WAV file."""
    import numpy as np

    # Ensure mono
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)

    # Convert to int16
    if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
        audio_array = (audio_array * 32767).clip(-32768, 32767).astype(np.int16)

    raw = audio_array.tobytes()

    # Resample if needed
    if sampling_rate != 16000:
        raw, _ = audioop.ratecv(raw, 2, 1, sampling_rate, 16000, None)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(raw)


def download_from_huggingface(languages: list[str], samples_per_lang: int):
    """Download samples from Mozilla Common Voice via HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package required. Install with:")
        print("  pip install datasets soundfile")
        sys.exit(1)

    manifest = []

    for lang in languages:
        config = LANG_CONFIGS.get(lang)
        if not config:
            print(f"Skipping unknown language: {lang}")
            continue

        lang_dir = os.path.join(UTTERANCES_DIR, lang)
        os.makedirs(lang_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Downloading {samples_per_lang} samples for {lang.upper()} (Common Voice)")
        print(f"{'='*60}")

        try:
            ds = load_dataset(
                "mozilla-foundation/common_voice_17_0",
                config,
                split=f"test[:{samples_per_lang}]",
            )
        except Exception as e:
            print(f"  Failed to load Common Voice for {lang}: {e}")
            print(f"  Trying google/fleurs fallback...")
            fleurs_map = {"hi": "hi_in", "ta": "ta_in", "te": "te_in", "bn": "bn_in"}
            try:
                ds = load_dataset(
                    "google/fleurs",
                    fleurs_map.get(lang, lang),
                    split=f"test[:{samples_per_lang}]",
                )
            except Exception as e2:
                print(f"  FLEURS also failed: {e2}")
                print(f"  Skipping {lang}")
                continue

        for i, sample in enumerate(ds):
            filename = f"{lang}_{i+1:03d}.wav"
            output_path = os.path.join(lang_dir, filename)

            # Extract audio and transcript
            audio = sample.get("audio", {})
            transcript = sample.get("sentence", sample.get("transcription", sample.get("transcript", "")))

            if isinstance(audio, dict) and "array" in audio:
                import numpy as np
                convert_to_wav16k(
                    np.array(audio["array"]),
                    audio.get("sampling_rate", 48000),
                    output_path,
                )
            elif isinstance(audio, dict) and "path" in audio:
                # Copy the file directly if it's already a path
                import shutil
                shutil.copy2(audio["path"], output_path)
            else:
                print(f"  Skipping sample {i+1}: unexpected audio format")
                continue

            manifest.append({
                "file": filename,
                "lang": lang,
                "reference_text": transcript,
                "source": "common_voice_17",
            })
            print(f"  [{i+1}/{samples_per_lang}] {filename}: {transcript[:60]}...")

        print(f"  Saved {len([m for m in manifest if m['lang'] == lang])} files to {lang_dir}")

    # Write manifest
    manifest_path = os.path.join(UTTERANCES_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest written: {manifest_path} ({len(manifest)} entries)")

    return manifest


def download_from_urls(languages: list[str], samples_per_lang: int):
    """Fallback: download from direct URLs (smaller free datasets)."""
    import urllib.request

    # OpenSLR datasets for Indic languages (free, no auth)
    # These are smaller but freely available
    OPENSLR_URLS = {
        "hi": "https://www.openslr.org/resources/103/hi_in_female.zip",
        "ta": "https://www.openslr.org/resources/65/ta_in_female.zip",
        "te": "https://www.openslr.org/resources/66/te_in_female.zip",
        "bn": "https://www.openslr.org/resources/37/bn_bd_female.zip",
    }

    manifest = []
    import tempfile
    import zipfile

    for lang in languages:
        url = OPENSLR_URLS.get(lang)
        if not url:
            print(f"No OpenSLR URL for {lang}, skipping")
            continue

        lang_dir = os.path.join(UTTERANCES_DIR, lang)
        os.makedirs(lang_dir, exist_ok=True)

        print(f"\nDownloading OpenSLR data for {lang.upper()}...")
        try:
            zip_path = os.path.join(tempfile.gettempdir(), f"openslr_{lang}.zip")
            urllib.request.urlretrieve(url, zip_path)

            with zipfile.ZipFile(zip_path) as zf:
                wav_files = [n for n in zf.namelist() if n.endswith(".wav")][:samples_per_lang]
                for i, wav_name in enumerate(wav_files):
                    filename = f"{lang}_{i+1:03d}.wav"
                    output_path = os.path.join(lang_dir, filename)
                    with zf.open(wav_name) as src, open(output_path, "wb") as dst:
                        dst.write(src.read())
                    manifest.append({
                        "file": filename,
                        "lang": lang,
                        "reference_text": "",
                        "source": "openslr",
                    })
                    print(f"  [{i+1}/{len(wav_files)}] {filename}")

            os.unlink(zip_path)
        except Exception as e:
            print(f"  Failed for {lang}: {e}")

    manifest_path = os.path.join(UTTERANCES_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest written: {manifest_path} ({len(manifest)} entries)")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Download Indic test audio")
    parser.add_argument("--languages", "-l", nargs="+", default=list(LANG_CONFIGS.keys()),
                        help="Languages to download (default: hi ta te bn)")
    parser.add_argument("--samples", "-n", type=int, default=5,
                        help="Samples per language (default: 5)")
    parser.add_argument("--source", choices=["huggingface", "openslr"], default="huggingface",
                        help="Data source (default: huggingface)")
    args = parser.parse_args()

    if args.source == "huggingface":
        download_from_huggingface(args.languages, args.samples)
    else:
        download_from_urls(args.languages, args.samples)


if __name__ == "__main__":
    main()
