"""
Generate Test Audio
===================
Generates synthetic test WAV files using Google TTS for benchmarking.
Creates 16kHz mono WAV files with known reference transcripts.

Usage:
    python scripts/generate_test_audio.py
    python scripts/generate_test_audio.py --samples 5
"""

import argparse
import json
import os
import struct
import wave
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTTERANCES_DIR = os.path.join(BASE_DIR, "test_utterances")

# Reference sentences for each language (real sentences, not lorem ipsum)
REFERENCE_TEXTS = {
    "hi": [
        "नमस्ते मैं आपकी कैसे मदद कर सकता हूँ",
        "कृपया अपना खाता नंबर बताइए",
        "आपका भुगतान सफलतापूर्वक हो गया है",
        "क्या आप अपना पासवर्ड बदलना चाहते हैं",
        "हमारी सेवा चौबीस घंटे उपलब्ध है",
        "आपकी शिकायत दर्ज कर ली गई है",
        "कृपया कुछ देर प्रतीक्षा करें",
        "आपका बैलेंस पाँच हज़ार रुपये है",
    ],
    "ta": [
        "வணக்கம் நான் உங்களுக்கு எப்படி உதவ முடியும்",
        "தயவுசெய்து உங்கள் கணக்கு எண்ணைக் கூறுங்கள்",
        "உங்கள் பணம் வெற்றிகரமாக செலுத்தப்பட்டது",
        "நீங்கள் உங்கள் கடவுச்சொல்லை மாற்ற விரும்புகிறீர்களா",
        "எங்கள் சேவை இருபத்தி நான்கு மணி நேரமும் கிடைக்கும்",
        "உங்கள் புகார் பதிவு செய்யப்பட்டது",
        "தயவுசெய்து சிறிது நேரம் காத்திருங்கள்",
        "உங்கள் இருப்பு ஐந்தாயிரம் ரூபாய்",
    ],
    "te": [
        "నమస్కారం నేను మీకు ఎలా సహాయం చేయగలను",
        "దయచేసి మీ ఖాతా నంబర్ చెప్పండి",
        "మీ చెల్లింపు విజయవంతంగా జరిగింది",
        "మీరు మీ పాస్వర్డ్ మార్చాలనుకుంటున్నారా",
        "మా సేవ ఇరవై నాలుగు గంటలు అందుబాటులో ఉంటుంది",
        "మీ ఫిర్యాదు నమోదు చేయబడింది",
        "దయచేసి కొంచెం సేపు వేచి ఉండండి",
        "మీ బ్యాలెన్స్ ఐదు వేల రూపాయలు",
    ],
    "bn": [
        "নমস্কার আমি আপনাকে কিভাবে সাহায্য করতে পারি",
        "দয়া করে আপনার অ্যাকাউন্ট নম্বর বলুন",
        "আপনার পেমেন্ট সফলভাবে হয়েছে",
        "আপনি কি আপনার পাসওয়ার্ড পরিবর্তন করতে চান",
        "আমাদের সেবা চব্বিশ ঘণ্টা উপলব্ধ",
        "আপনার অভিযোগ নথিভুক্ত করা হয়েছে",
        "দয়া করে কিছুক্ষণ অপেক্ষা করুন",
        "আপনার ব্যালেন্স পাঁচ হাজার টাকা",
    ],
}


def text_to_wav_gtts(text: str, lang: str, output_path: str) -> bool:
    """Generate WAV from text using Google TTS (requires internet)."""
    try:
        from gtts import gTTS
        import io
        import tempfile
        import subprocess

        tts = gTTS(text=text, lang=lang)
        mp3_path = output_path.replace(".wav", ".mp3")
        tts.save(mp3_path)

        # Convert mp3 to 16kHz mono WAV using ffmpeg if available
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", output_path],
            capture_output=True, timeout=30,
        )
        os.unlink(mp3_path)
        return result.returncode == 0
    except Exception:
        return False


def text_to_wav_synth(text: str, lang: str, output_path: str):
    """Generate a synthetic WAV with a tone pattern as placeholder.

    Each character maps to a frequency, creating a unique audio fingerprint
    per utterance. Duration scales with text length.
    """
    sample_rate = 16000
    # ~0.15s per character, min 1s
    duration_s = max(1.0, len(text) * 0.15)
    n_samples = int(sample_rate * duration_s)

    samples = []
    chars = list(text)
    segment_len = n_samples // max(len(chars), 1)

    for i, ch in enumerate(chars):
        # Map character ordinal to frequency 200-800Hz
        freq = 200 + (ord(ch) % 600)
        for j in range(segment_len):
            t = j / sample_rate
            val = int(16000 * math.sin(2 * math.pi * freq * t))
            val = max(-32768, min(32767, val))
            samples.append(val)

    # Pad to exact length
    while len(samples) < n_samples:
        samples.append(0)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def main():
    parser = argparse.ArgumentParser(description="Generate test audio for benchmarking")
    parser.add_argument("--languages", "-l", nargs="+", default=["hi", "ta", "te", "bn"])
    parser.add_argument("--samples", "-n", type=int, default=5)
    parser.add_argument("--method", choices=["gtts", "synth"], default="gtts",
                        help="gtts=Google TTS (realistic), synth=tone patterns (offline)")
    args = parser.parse_args()

    manifest = []
    use_gtts = args.method == "gtts"

    # Check if gtts and ffmpeg are available
    if use_gtts:
        try:
            from gtts import gTTS
            import subprocess
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except Exception:
            print("gTTS or ffmpeg not available, falling back to synth method")
            use_gtts = False

    for lang in args.languages:
        texts = REFERENCE_TEXTS.get(lang, [])
        if not texts:
            print(f"No reference texts for {lang}, skipping")
            continue

        lang_dir = os.path.join(UTTERANCES_DIR, lang)
        os.makedirs(lang_dir, exist_ok=True)
        n = min(args.samples, len(texts))

        print(f"\n{'='*60}")
        print(f"  Generating {n} samples for {lang.upper()}")
        print(f"{'='*60}")

        for i in range(n):
            filename = f"{lang}_{i+1:03d}.wav"
            output_path = os.path.join(lang_dir, filename)
            text = texts[i]

            if use_gtts:
                ok = text_to_wav_gtts(text, lang, output_path)
                if not ok:
                    print(f"  gTTS failed for {filename}, using synth")
                    text_to_wav_synth(text, lang, output_path)
            else:
                text_to_wav_synth(text, lang, output_path)

            manifest.append({
                "file": filename,
                "lang": lang,
                "reference_text": text,
                "source": "gtts" if use_gtts else "synth",
            })
            print(f"  [{i+1}/{n}] {filename}: {text[:60]}...")

    manifest_path = os.path.join(UTTERANCES_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest written: {manifest_path} ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
