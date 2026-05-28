"""
contextual.py
=============
Implements Anthropic's Contextual Retrieval approach.

Each chunk is enriched with a short LLM-generated description of *where it
sits within the source document* before being embedded.  This prefix is
prepended to the chunk text at embedding time (via Chunk.content_for_embedding)
so the vector representation captures document-level context, not just the
local text.

Reference: https://www.anthropic.com/news/contextual-retrieval

Classes:
    ContextualEnricher — enriches a list of Chunk objects in-place

Usage:
    enricher = ContextualEnricher()
    chunks   = enricher.enrich(chunks, full_document_text)
"""

from __future__ import annotations

import time
from typing import List, Optional

import anthropic

from chunkers import Chunk


# ── Prompt template ────────────────────────────────────────────────────────────

_CONTEXT_PROMPT = """\
<document>
{document}
</document>

Here is a chunk from that document:
<chunk>
{chunk}
</chunk>

This chunk comes from the following location in the document:
<location>{location}</location>

Write 2-3 sentences that situate this chunk within the document for improved \
search retrieval. Cover:
  • The section hierarchy it belongs to (use the location above)
  • The main concept or data point it contains
  • How it relates to the document's overall purpose

Reply with ONLY the context sentences — no preamble, no labels."""


# ── Enricher ───────────────────────────────────────────────────────────────────

class ContextualEnricher:
    """
    Prepends each chunk with LLM-generated context before it is embedded.

    Design notes:
    - Uses Claude Haiku by default — fast and cheap for short generation tasks.
    - Very long documents are trimmed (keeping start + end) to stay within the
      context window while preserving the most structurally important parts.
    - A small inter-request delay avoids rate-limit errors on large batches.

    Args:
        model           — Anthropic model for context generation (default: Haiku)
        max_doc_words   — max words of source doc passed to the prompt (default 6000)
        request_delay   — seconds to sleep between API calls (default 0.15)
        fallback_to_breadcrumb — if True, use the chunk's breadcrumb string as
                                 fallback context when the API call fails
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_doc_words: int = 6_000,
        request_delay: float = 0.15,
        fallback_to_breadcrumb: bool = True,
    ):
        self.model = model
        self.max_doc_words = max_doc_words
        self.request_delay = request_delay
        self.fallback_to_breadcrumb = fallback_to_breadcrumb
        self._client = anthropic.Anthropic()

    # ── Public API ─────────────────────────────────────────────────────────────

    def enrich(
        self,
        chunks: List[Chunk],
        document: str,
        verbose: bool = True,
    ) -> List[Chunk]:
        """
        Enrich *chunks* with contextual prefixes derived from *document*.

        Mutates each Chunk's ``contextual_prefix`` field in-place and also
        returns the list (for convenience in pipelines).

        Args:
            chunks   — output of any BaseChunker.chunk() call
            document — the full source document text
            verbose  — print progress to stdout

        Returns:
            The same list of Chunk objects, now with contextual_prefix set.
        """
        trimmed_doc = self._trim_document(document)
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            if verbose:
                level_tag = f"[{chunk.level}]"
                sec_tag = f" § {chunk.section_title}" if chunk.section_title else ""
                print(f"  Enriching {i+1}/{total} {level_tag}{sec_tag} ...")

            try:
                prefix = self._generate_context(chunk, trimmed_doc)
                chunk.contextual_prefix = prefix
            except Exception as exc:
                if verbose:
                    print(f"    ⚠ API error for {chunk.chunk_id}: {exc}")
                if self.fallback_to_breadcrumb:
                    chunk.contextual_prefix = chunk.parent_section or chunk.section_title or ""
                # else leave contextual_prefix as None (chunk embeds without prefix)

            if self.request_delay and i < total - 1:
                time.sleep(self.request_delay)

        if verbose:
            print(f"  ✓ Enrichment complete ({total} chunks)")

        return chunks

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _generate_context(self, chunk: Chunk, trimmed_document: str) -> str:
        """Call the LLM and return the generated context string."""
        # Build a human-readable location string from the chunk's breadcrumb.
        # e.g.  "2. Pricing > 2.1 Base Unit Pricing"
        # Falls back to section_title or "Unknown" if neither is set.
        location = (
            chunk.parent_section
            or chunk.section_title
            or "Unknown"
        )
        prompt = _CONTEXT_PROMPT.format(
            document=trimmed_document,
            chunk=chunk.text,
            location=location,
        )
        response = self._client.messages.create(
            model=self.model,
            max_tokens=220,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def _trim_document(self, document: str) -> str:
        """
        Keep document within max_doc_words by retaining the first 70 % and
        last 30 % of words — this preserves the intro (scope / purpose) and
        the tail (signatures / appendices) which are often useful as context.
        """
        words = document.split()
        if len(words) <= self.max_doc_words:
            return document

        keep_start = int(self.max_doc_words * 0.70)
        keep_end = int(self.max_doc_words * 0.30)

        head = " ".join(words[:keep_start])
        tail = " ".join(words[-keep_end:])
        return head + "\n\n[... document trimmed for context window ...]\n\n" + tail
