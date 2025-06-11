# Extract & Transcribe

Extract audio from video files, and create trascripts using `faster_whisper`.

## Installation

```bash
poetry install
```

## Run

### 1. Extract audio from video

1. Create `videos` directory.
2. Place all videos in videos/
3. Create `mp3` directory (if it doesn't already exist)
4. Run `extract_audio.sh`

### 2. Create transcripts

```bash
$ poetry run python transcribe.py
```

Created with much love ❤️
