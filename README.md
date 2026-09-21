# Extract & Transcribe

Extract audio from video files and transcribe them locally with Whisper **large-v3**.
On Apple Silicon Macs, MLX runs the model on the GPU through Metal. Other systems
use `faster-whisper` on the CPU.

## Installation

Use Poetry 2 and Python 3.11–3.13 with the current locked dependencies.

On an **Apple Silicon Mac host**, using native ARM64 Python, run:

```bash
brew install ffmpeg
poetry install -E mac
```

The `mac` extra installs `mlx-whisper` only on macOS ARM64. Run GPU transcription
directly on the Mac, outside Docker. For CPU use on other systems, run
`poetry install` instead. The project uses Poetry's non-package mode, so
`--no-root` is no longer necessary.

## Run

### 1. Extract audio from video

1. Create `videos` directory.
2. Place all videos in videos/
3. Create `mp3` directory (if it doesn't already exist)
4. Run `extract_audio.sh`

### 2. Create transcripts

```bash
poetry run python transcribe.py
```

On Apple Silicon this automatically selects MLX. To explicitly require the GPU:

```bash
poetry run python transcribe.py --backend mlx
```

The script prints `Backend: MLX / Apple GPU (Metal)` after checking GPU access.
The first transcription downloads a separate MLX copy of the full large-v3 model
(several GB); later runs reuse the cached weights. Allow sufficient free memory
for the model and audio. GPU inference can improve speed, but the actual speed
depends on your Mac and recording length. MLX uses FP16 and its own decoding
implementation, so transcripts may differ from the CPU int8 backend.

Useful options:

```bash
# Specify a language (otherwise detected automatically).
poetry run python transcribe.py --language el

# Reprocess existing files with the GPU, saving results separately for comparison.
poetry run python transcribe.py --backend mlx --output-dir transcripts/mlx --force

# Explicit CPU mode; includes voice activity detection to filter silence.
poetry run python transcribe.py --backend cpu
```

Per-file entries in `LANGUAGE_OVERRIDES` take precedence over `--language`.
Use `--input-dir` to select another MP3 directory. Completed transcripts are
written atomically; failures leave any existing transcript intact. Existing
nonempty transcripts are skipped by default. Use `--force` to regenerate files
that may have been left incomplete by the old script, or after changing settings.
MLX uses Whisper's built-in silence detection; the separate Silero VAD filter
is enabled only for the CPU backend.

Verify the workflow without downloading models:

```bash
python3 -m unittest discover -s tests -v
```

References: [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
and the [MLX large-v3 weights](https://huggingface.co/mlx-community/whisper-large-v3-mlx).

Created with much love ❤️
