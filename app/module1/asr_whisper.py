from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_ASR_MODEL = "openai/whisper-large-v3"
DEFAULT_LANGUAGE = "turkish"
DEFAULT_TASK = "transcribe"

_PIPELINE_CACHE: dict[str, object] = {}


@dataclass
class AsrTranscriptionResult:
    text: str
    model_id: str
    backend: str = "transformers-whisper"
    language: str = DEFAULT_LANGUAGE


def _load_pipeline(model_id: str):
    if model_id in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[model_id]

    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Whisper ASR için `transformers`, `torch` ve ilgili ses bağımlılıkları gerekli."
        ) from exc

    device = 0 if torch.cuda.is_available() else -1
    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        dtype=model_dtype,
        device=device,
    )
    _PIPELINE_CACHE[model_id] = pipe
    return pipe


def transcribe_audio_file(
    audio_path: str | Path,
    model_id: str = DEFAULT_ASR_MODEL,
    language: str = DEFAULT_LANGUAGE,
) -> AsrTranscriptionResult:
    pipe = _load_pipeline(model_id)
    result = pipe(
        str(audio_path),
        generate_kwargs={"language": language, "task": DEFAULT_TASK},
    )
    text = str(result.get("text", "")).strip()
    return AsrTranscriptionResult(
        text=text,
        model_id=model_id,
        language=language,
    )
