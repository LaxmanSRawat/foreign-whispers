"""HTTP-agnostic service wrapping translation engine functions."""

import asyncio
import copy
import inspect
import pathlib
import re
from pathlib import Path

from api.src.services import translation_engine as _te


def _split_sentences(text: str) -> list[str]:
    """Split translated text into sentences on . ? ! boundaries."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def download_and_install_package(from_code: str, to_code: str):
    return _te.download_and_install_package(from_code, to_code)


def translate_sentence(text: str, from_code: str, to_code: str):
    return _te.translate_sentence(text, from_code, to_code)


class TranslationService:
    """Thin wrapper around argostranslate-based translation.

    Takes *ui_dir* via constructor so the caller controls file paths.
    """

    def __init__(self, ui_dir: Path) -> None:
        self.ui_dir = ui_dir

    def install_language_pack(self, from_code: str, to_code: str) -> None:
        """Download and install the Argos Translate language pack."""
        download_and_install_package(from_code, to_code)

    def translate_sentence(self, text: str, from_code: str, to_code: str) -> str:
        """Translate a single sentence."""
        return translate_sentence(text, from_code, to_code)

    def translate_transcript(self, transcript: dict, from_code: str, to_code: str) -> dict:
        """Translate all segments and full text in a transcript dict.

        Translates the full paragraph as a single unit so argostranslate has
        sentence-level context (e.g. "60 Minutes Overtime" vs "over time"),
        then splits the result back to individual segments. Falls back to
        per-segment translation when the sentence count doesn't match.

        Returns a deep copy; the original is not mutated.
        """
        result = copy.deepcopy(transcript)
        segments = result.get("segments", [])

        full_source = " ".join(s["text"].strip() for s in segments)
        full_translation = translate_sentence(full_source, from_code, to_code)
        result["text"] = full_translation
        result["language"] = to_code

        translated_parts = _split_sentences(full_translation)
        if len(translated_parts) == len(segments):
            for seg, text in zip(segments, translated_parts):
                seg["text"] = text
        else:
            for seg in segments:
                seg["text"] = translate_sentence(seg["text"], from_code, to_code)

        return result

    def rerank_for_duration(
        self,
        en_transcript: dict,
        es_transcript: dict,
        from_code: str = "en",
        to_code: str = "es",
    ) -> dict:
        """Re-rank translated segments that exceed their duration budget.

        For each segment where decide_action() returns REQUEST_SHORTER, calls
        get_shorter_translations() to produce shorter alternatives and picks
        the best fit.  Returns a deep copy of es_transcript; original is never
        mutated.

        Note: get_shorter_translations() is a student assignment stub that
        currently returns an empty list — so this method is a no-op until
        the stub is implemented.
        """
        import copy
        from foreign_whispers.alignment import AlignAction, compute_segment_metrics, decide_action
        from foreign_whispers.reranking import get_shorter_translations

        def _resolve_candidates(candidates_or_awaitable):
            if not inspect.isawaitable(candidates_or_awaitable):
                return candidates_or_awaitable
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(candidates_or_awaitable)
            # This service method is synchronous; if it is called from an
            # async context, leave the original translation untouched rather
            # than trying to nest event loops.
            return []

        result = copy.deepcopy(es_transcript)
        metrics = compute_segment_metrics(en_transcript, es_transcript)

        for m in metrics:
            if decide_action(m) != AlignAction.REQUEST_SHORTER:
                continue
            segs = es_transcript.get("segments", [])
            prev = segs[m.index - 1]["text"] if m.index > 0 else ""
            nxt  = segs[m.index + 1]["text"] if m.index < len(segs) - 1 else ""

            candidates = _resolve_candidates(
                get_shorter_translations(
                    source_text       = m.source_text,
                    baseline_es       = m.translated_text,
                    target_duration_s = m.source_duration_s,
                    context_prev      = prev,
                    context_next      = nxt,
                )
            )

            if candidates:
                best = min(
                    candidates,
                    key=lambda c: abs(len(c.text) / 15.0 - m.source_duration_s),
                )
                result["segments"][m.index]["text"] = best.text

        return result

    @staticmethod
    def title_for_video_id(video_id: str, search_dir: pathlib.Path) -> str | None:
        """Find a title by scanning *search_dir* for JSON files."""
        for f in search_dir.glob("*.json"):
            return f.stem
        return None
