from pathlib import Path
from faster_whisper import WhisperModel

# Model configuration
MODEL_SIZE = "large-v3"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# Paths
AUDIO_DIR = Path("mp3")
OUTPUT_DIR = Path("transcripts")
OUTPUT_DIR.mkdir(exist_ok=True)

# Optional: map specific filenames to forced languages
# Use ISO 639-1 codes: "en" (English), "el" (Greek), "es" (Spanish), etc.
LANGUAGE_OVERRIDES = {
}


def transcript_exists_and_not_empty(transcript_path: Path) -> bool:
    return transcript_path.exists() and transcript_path.stat().st_size > 0


def transcribe_all():
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    for mp3_file in AUDIO_DIR.glob("*.mp3"):
        transcript_path = OUTPUT_DIR / f"{mp3_file.stem}.txt"
        if transcript_exists_and_not_empty(transcript_path):
            print(f"Skipping {mp3_file.name} (already transcribed).")
            continue

        language = LANGUAGE_OVERRIDES.get(mp3_file.name)
        print(f"Transcribing {mp3_file.name} with language={language or 'auto-detect'}...")

        segments, _ = model.transcribe(str(mp3_file), language=language)

        with open(transcript_path, "w", encoding="utf-8") as f:
            for segment in segments:
                f.write(segment.text + "\n")

        print(f"Saved transcript to {transcript_path}")


if __name__ == "__main__":
    if not AUDIO_DIR.exists():
        print(f"Folder '{AUDIO_DIR}' not found. Please create it and add .mp3 files.")
    else:
        transcribe_all()
