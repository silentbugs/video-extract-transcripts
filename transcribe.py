import argparse
import os
import platform
import shutil
import tempfile
from pathlib import Path

MODEL_SIZE = "large-v3"
MLX_MODEL = "mlx-community/whisper-large-v3-mlx"
AUDIO_DIR = Path("mp3")
OUTPUT_DIR = Path("transcripts")

# Optional per-file overrides, using ISO language codes such as "en" or "el".
LANGUAGE_OVERRIDES = {}


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def create_transcriber(backend: str):
    if backend == "auto":
        backend = "mlx" if is_apple_silicon() else "cpu"

    if backend == "mlx":
        if not is_apple_silicon():
            raise RuntimeError(
                "MLX GPU transcription requires native ARM64 Python on an Apple "
                "Silicon Mac. Run this on your Mac host, outside Docker."
            )
        try:
            import mlx.core as mx
            import mlx_whisper
        except ImportError as exc:
            raise RuntimeError(
                "Install the Mac dependencies on your host with "
                "`poetry install -E mac`, or select `--backend cpu`."
            ) from exc
        if not mx.metal.is_available():
            raise RuntimeError("MLX cannot access Metal. Run on your Mac host with GPU access.")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("MLX Whisper requires FFmpeg: run `brew install ffmpeg` on your Mac.")
        mx.set_default_device(mx.gpu)
        print(f"Backend: MLX / Apple GPU (Metal); model: {MLX_MODEL}", flush=True)

        def transcribe(path, language):
            result = mlx_whisper.transcribe(
                str(path), path_or_hf_repo=MLX_MODEL, language=language,
                task="transcribe", fp16=True, verbose=False,
            )
            return (segment["text"] for segment in result["segments"])

        return transcribe

    from faster_whisper import WhisperModel

    print(f"Backend: faster-whisper / CPU (int8); model: {MODEL_SIZE}", flush=True)
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    def transcribe(path, language):
        segments, _ = model.transcribe(str(path), language=language, vad_filter=True)
        return (segment.text for segment in segments)

    return transcribe


def write_transcript(path: Path, texts):
    """Publish only a complete transcript, preserving any prior file on failure."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as output:
            temporary_path = Path(output.name)
            for text in texts:
                output.write(text.strip() + "\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def transcribe_all(
    backend="auto", language=None, force=False,
    audio_dir=AUDIO_DIR, output_dir=OUTPUT_DIR,
):
    if not audio_dir.is_dir():
        raise RuntimeError(f"Folder '{audio_dir}' not found. Create it and add .mp3 files.")
    audio_files = sorted(
        path for path in audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp3"
    )
    if not audio_files:
        print(f"No MP3 files found in {audio_dir}.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    transcribe = None
    for audio_path in audio_files:
        transcript_path = output_dir / f"{audio_path.stem}.txt"
        if not force and transcript_path.is_file() and transcript_path.stat().st_size > 0:
            print(f"Skipping {audio_path.name} (already transcribed; use --force to rerun).")
            continue
        if transcribe is None:
            transcribe = create_transcriber(backend)
        selected_language = LANGUAGE_OVERRIDES.get(audio_path.name, language)
        print(
            f"Transcribing {audio_path.name} "
            f"with language={selected_language or 'auto-detect'}...",
            flush=True,
        )
        write_transcript(transcript_path, transcribe(audio_path, selected_language))
        print(f"Saved transcript to {transcript_path}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe MP3 files using Whisper large-v3.")
    parser.add_argument(
        "--backend", choices=("auto", "mlx", "cpu"), default="auto",
        help="auto selects MLX on Apple Silicon and CPU elsewhere",
    )
    parser.add_argument("--language", help="Language code, e.g. en or el; default: auto-detect")
    parser.add_argument("--force", action="store_true", help="Replace existing transcripts")
    parser.add_argument("--input-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    try:
        transcribe_all(
            backend=args.backend, language=args.language, force=args.force,
            audio_dir=args.input_dir, output_dir=args.output_dir,
        )
    except (RuntimeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
