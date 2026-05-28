"""
chunkers.py
===========
Modular document chunking strategies for the RAG pipeline.

Classes:
    Chunk               — dataclass representing a single chunk with metadata
    BaseChunker         — abstract base; all chunkers implement .chunk()
    FixedSizeChunker    — baseline: splits by token count with overlap
    HierarchicalChunker — splits by document structure (headers → paragraphs → sentences)

Usage:
    chunker = HierarchicalChunker()
    chunks  = chunker.chunk(document_text, doc_id="supplier_agreement_v2")
"""

from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

# ── Optional tiktoken for accurate token counting ─────────────────────────────
try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _ENCODER = None


def _token_count(text: str) -> int:
    """Approximate token count. Uses tiktoken if available, falls back to word count."""
    if _ENCODER:
        return len(_ENCODER.encode(text))
    return int(len(text.split()) * 1.35)  # rough 1.35 tokens per word


# ── Chunk dataclass ────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A single chunk of a document, with structural metadata.

    Attributes:
        text              — original chunk text (used for display / citation)
        doc_id            — source document identifier
        chunk_id          — unique chunk identifier
        level             — granularity: 'section' | 'paragraph' | 'sentence' | 'fixed'
        section_title     — title of the immediate section this chunk belongs to
        parent_section    — full breadcrumb path, e.g. "Contract > 2. Pricing > 2.1 Terms"
        metadata          — arbitrary extra fields (section level, para index, etc.)
        contextual_prefix — LLM-generated context added by ContextualEnricher before embedding
    """
    text: str
    doc_id: str
    chunk_id: str
    level: str
    section_title: Optional[str] = None
    parent_section: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    contextual_prefix: Optional[str] = None

    @property
    def content_for_embedding(self) -> str:
        """
        Text that gets embedded into the vector store.
        If a contextual prefix has been generated, it is prepended so the
        embedding captures both the local content and its document-level context.
        """
        if self.contextual_prefix:
            return f"{self.contextual_prefix}\n\n{self.text}"
        return self.text

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return (
            f"Chunk(id={self.chunk_id!r}, level={self.level!r}, "
            f"section={self.section_title!r}, tokens={_token_count(self.text)}, "
            f"text={preview!r}...)"
        )


# ── Base class ─────────────────────────────────────────────────────────────────

class BaseChunker(ABC):
    """Abstract base for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Split *text* into a list of Chunk objects.

        Args:
            text   — full document text
            doc_id — identifier used in chunk IDs and metadata

        Returns:
            Ordered list of Chunk objects.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ── Fixed-size chunker ─────────────────────────────────────────────────────────

class FixedSizeChunker(BaseChunker):
    """
    Baseline chunker: splits by token count with a sliding overlap window.

    Best for: quick prototyping, documents without clear structure.
    Weakness: cuts across sentence / paragraph boundaries.

    Args:
        chunk_size — target tokens per chunk (default 512)
        overlap    — tokens of overlap between consecutive chunks (default 50)
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if _ENCODER:
            tokens = _ENCODER.encode(text)
            decode = lambda toks: _ENCODER.decode(toks)
        else:
            # word-level fallback
            tokens = text.split()
            decode = lambda toks: " ".join(toks)

        chunks: List[Chunk] = []
        start = 0
        idx = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_text = decode(tokens[start:end])
            chunks.append(Chunk(
                text=chunk_text,
                doc_id=doc_id,
                chunk_id=f"{doc_id}_fixed_{idx}",
                level="fixed",
                metadata={"token_start": start, "token_end": end},
            ))
            start += self.chunk_size - self.overlap
            idx += 1

        return chunks


# ── Hierarchical chunker ───────────────────────────────────────────────────────

# Ordered list of (regex_pattern, heading_level).
# Evaluated top-to-bottom; first match wins.
_HEADER_RULES: List[tuple] = [
    (r"^#{1}\s+(.+)$",         1),   # Markdown H1
    (r"^#{2}\s+(.+)$",         2),   # Markdown H2
    (r"^#{3}\s+(.+)$",         3),   # Markdown H3
    (r"^#{4}\s+(.+)$",         4),   # Markdown H4
    (r"^(\d+\.\s+.+)$",        2),   # Numbered: "1. Section Title"
    (r"^(\d+\.\d+\s+.+)$",     3),   # Sub-numbered: "1.1 Sub-section"
    (r"^([A-Z][A-Z\s]{4,80})$", 1),  # ALL CAPS heading (min 5 chars)
]


@dataclass
class _Section:
    title: str
    level: int
    content: str
    breadcrumb: List[str]  # ancestor titles, from root to parent


class HierarchicalChunker(BaseChunker):
    """
    Structure-aware chunker that preserves document hierarchy.

    Pipeline:
        1. Detect headers → split into sections (each section carries a breadcrumb)
        2. Within each section, split by double-newline paragraphs
        3. If a paragraph exceeds *max_paragraph_tokens*, slide a sentence window over it

    The breadcrumb (e.g. "Contract > 2. Pricing > 2.1 Early Payment") is stored
    in every chunk, giving the LLM rich context about where the text came from.

    Args:
        max_paragraph_tokens — paragraphs larger than this get sentence-windowed (default 350)
        sentence_window_size — number of sentences per sentence-level chunk (default 3)
        sentence_overlap     — sentences of overlap between windows (default 1)
    """

    def __init__(
        self,
        max_paragraph_tokens: int = 350,
        sentence_window_size: int = 3,
        sentence_overlap: int = 1,
    ):
        self.max_paragraph_tokens = max_paragraph_tokens
        self.sentence_window_size = sentence_window_size
        self.sentence_overlap = max(0, sentence_overlap)

    # ── Public API ─────────────────────────────────────────────────────────────

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        sections = self._parse_sections(text)
        all_chunks: List[Chunk] = []

        for sec_idx, section in enumerate(sections):
            para_chunks = self._section_to_chunks(section, doc_id, sec_idx)
            all_chunks.extend(para_chunks)

        return all_chunks

    # ── Section parsing ────────────────────────────────────────────────────────

    def _parse_sections(self, text: str) -> List[_Section]:
        """
        Walk through lines, detect headers, and accumulate content per section.
        Returns a flat list of _Section objects (hierarchy is captured in breadcrumbs).
        """
        lines = text.splitlines()
        sections: List[_Section] = []

        # Sentinel: content before the first header lands in an "Introduction" section
        current = _Section(title="Introduction", level=0, content="", breadcrumb=[])
        # Stack tracks open ancestors: list of (title, level)
        ancestor_stack: List[tuple] = []

        for line in lines:
            matched_level, matched_title = self._match_header(line)

            if matched_level is not None:
                # Flush current section
                body = current.content.strip()
                if body:
                    sections.append(current)

                # Pop ancestors that are same level or deeper
                while ancestor_stack and ancestor_stack[-1][1] >= matched_level:
                    ancestor_stack.pop()

                breadcrumb = [a[0] for a in ancestor_stack]
                current = _Section(
                    title=matched_title,
                    level=matched_level,
                    content="",
                    breadcrumb=breadcrumb,
                )
                ancestor_stack.append((matched_title, matched_level))
            else:
                current.content += line + "\n"

        # Flush last section
        if current.content.strip():
            sections.append(current)

        return sections

    @staticmethod
    def _match_header(line: str):
        """Return (level, title) if line is a header, else (None, None)."""
        stripped = line.strip()
        for pattern, level in _HEADER_RULES:
            m = re.match(pattern, stripped)
            if m:
                return level, m.group(1).strip()
        return None, None

    # ── Section → chunks ───────────────────────────────────────────────────────

    def _section_to_chunks(
        self, section: _Section, doc_id: str, sec_idx: int
    ) -> List[Chunk]:
        """Split one section into paragraph- or sentence-level chunks."""
        breadcrumb_parts = section.breadcrumb + [section.title]
        breadcrumb_str = " > ".join(breadcrumb_parts)

        # Split on blank lines to get paragraphs
        raw_paras = re.split(r"\n\s*\n", section.content)
        paragraphs = [p.strip() for p in raw_paras if p.strip()]

        chunks: List[Chunk] = []

        for para_idx, para in enumerate(paragraphs):
            if _token_count(para) <= self.max_paragraph_tokens:
                # Paragraph fits as-is
                chunks.append(Chunk(
                    text=para,
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_s{sec_idx}_p{para_idx}",
                    level="paragraph",
                    section_title=section.title,
                    parent_section=breadcrumb_str,
                    metadata={
                        "section_level": section.level,
                        "breadcrumb": breadcrumb_str,
                        "para_index": para_idx,
                    },
                ))
            else:
                # Paragraph too large — slide a sentence window over it
                sent_chunks = self._sentence_window_chunks(
                    para, doc_id, section, sec_idx, para_idx, breadcrumb_str
                )
                chunks.extend(sent_chunks)

        return chunks

    # ── Sentence windowing ─────────────────────────────────────────────────────

    def _sentence_window_chunks(
        self,
        text: str,
        doc_id: str,
        section: _Section,
        sec_idx: int,
        para_idx: int,
        breadcrumb_str: str,
    ) -> List[Chunk]:
        """
        Slide a window of *sentence_window_size* sentences over *text*,
        stepping by (window_size - overlap) each time.
        """
        # Sentence split on .!? followed by whitespace or end-of-string
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

        if not sentences:
            return []

        step = max(1, self.sentence_window_size - self.sentence_overlap)
        chunks: List[Chunk] = []
        win_idx = 0

        for start in range(0, len(sentences), step):
            window = sentences[start : start + self.sentence_window_size]
            if not window:
                break
            chunk_text = " ".join(window)
            chunks.append(Chunk(
                text=chunk_text,
                doc_id=doc_id,
                chunk_id=f"{doc_id}_s{sec_idx}_p{para_idx}_w{win_idx}",
                level="sentence",
                section_title=section.title,
                parent_section=breadcrumb_str,
                metadata={
                    "section_level": section.level,
                    "breadcrumb": breadcrumb_str,
                    "para_index": para_idx,
                    "sentence_start": start,
                },
            ))
            win_idx += 1

        return chunks
