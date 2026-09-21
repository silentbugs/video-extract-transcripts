import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import transcribe


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.audio = self.root / "audio"
        self.output = self.root / "output"
        self.audio.mkdir()
        self.output.mkdir()
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def test_failed_transcription_preserves_previous_result_and_cleans_temporary_file(self):
        path = self.output / "recording.txt"

        def failing_segments():
            yield "partial transcript"
            raise RuntimeError("inference failed")

        for existing in (False, True):
            with self.subTest(existing=existing):
                if existing:
                    path.write_text("previous complete transcript", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    transcribe.write_transcript(path, failing_segments())
                if existing:
                    self.assertEqual(path.read_text(), "previous complete transcript")
                else:
                    self.assertFalse(path.exists())
                self.assertEqual(list(self.output.glob("*.tmp")), [])

    def test_success_publishes_complete_unicode_transcript(self):
        path = self.output / "recording.txt"
        transcribe.write_transcript(path, iter([" Γεια σου", " κόσμε "]))
        self.assertEqual(path.read_text(encoding="utf-8"), "Γεια σου\nκόσμε\n")
        self.assertEqual(list(self.output.iterdir()), [path])

    def test_existing_transcripts_skip_model_loading(self):
        (self.audio / "recording.mp3").touch()
        (self.output / "recording.txt").write_text("complete")
        with patch.object(transcribe, "create_transcriber") as create:
            transcribe.transcribe_all(audio_dir=self.audio, output_dir=self.output)
        create.assert_not_called()

    def test_force_and_language_override_with_one_model_for_batch(self):
        for name in ("one.MP3", "two.mp3"):
            (self.audio / name).touch()
            (self.output / f"{Path(name).stem}.txt").write_text("old")
        inference = Mock(return_value=["new transcript"])
        with patch.object(transcribe, "create_transcriber", return_value=inference) as create:
            with patch.dict(transcribe.LANGUAGE_OVERRIDES, {"one.MP3": "el"}, clear=True):
                transcribe.transcribe_all(
                    backend="mlx", language="en", force=True,
                    audio_dir=self.audio, output_dir=self.output,
                )
        create.assert_called_once_with("mlx")
        self.assertEqual([call.args[1] for call in inference.call_args_list], ["el", "en"])
        self.assertEqual((self.output / "one.txt").read_text(), "new transcript\n")

    def test_mlx_rejects_container_without_importing_dependencies(self):
        with patch.object(transcribe, "is_apple_silicon", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Mac host, outside Docker"):
                transcribe.create_transcriber("mlx")

    def test_auto_cpu_transcribes_with_silence_filter(self):
        faster_whisper = ModuleType("faster_whisper")
        model = Mock()
        model.transcribe.return_value = (iter([SimpleNamespace(text="hello")]), None)
        faster_whisper.WhisperModel = Mock(return_value=model)
        with patch.dict("sys.modules", {"faster_whisper": faster_whisper}):
            with patch.object(transcribe, "is_apple_silicon", return_value=False):
                inference = transcribe.create_transcriber("auto")
                self.assertEqual(list(inference(Path("sample.mp3"), None)), ["hello"])
        faster_whisper.WhisperModel.assert_called_once_with(
            "large-v3", device="cpu", compute_type="int8",
        )
        model.transcribe.assert_called_once_with("sample.mp3", language=None, vad_filter=True)

    def test_auto_mlx_selects_gpu_and_full_large_v3(self):
        mlx = ModuleType("mlx")
        core = ModuleType("mlx.core")
        core.gpu = object()
        core.metal = Mock()
        core.metal.is_available.return_value = True
        core.set_default_device = Mock()
        mlx.core = core
        whisper = ModuleType("mlx_whisper")
        whisper.transcribe = Mock(return_value={"segments": [{"text": "hello"}]})
        modules = {"mlx": mlx, "mlx.core": core, "mlx_whisper": whisper}
        with patch.dict("sys.modules", modules):
            with patch.object(transcribe, "is_apple_silicon", return_value=True):
                with patch.object(transcribe.shutil, "which", return_value="/usr/bin/ffmpeg"):
                    inference = transcribe.create_transcriber("auto")
                    self.assertEqual(list(inference(Path("sample.mp3"), "en")), ["hello"])
                core.metal.is_available.return_value = False
                with self.assertRaisesRegex(RuntimeError, "cannot access Metal"):
                    transcribe.create_transcriber("mlx")
        core.set_default_device.assert_called_once_with(core.gpu)
        whisper.transcribe.assert_called_once_with(
            "sample.mp3", path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
            language="en", task="transcribe", fp16=True, verbose=False,
        )


if __name__ == "__main__":
    unittest.main()
