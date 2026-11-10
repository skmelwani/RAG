"""Split cleaned pages into token-sized, overlapping chunks.

Strategy:
1. Break a page into "units": paragraphs, or sentences if a paragraph is too
   long, or hard token windows if even a sentence is too long.
2. Pack units into a chunk until the next unit would exceed chunk_size tokens.
3. Start the next chunk with the last few units of the previous one (overlap),
   so a fact sitting on a boundary appears whole in at least one chunk.

Chunks never cross pages, so every chunk cites exactly one page.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from transformers import PreTrainedTokenizerBase

from .models import Chunk, Page

# Split after . ! or ? when followed by whitespace and an uppercase letter,
# digit, bracket or quote. Simple, and it will wrongly split after "e.g. The".
# Good enough for chunking: a slightly wrong split only moves a boundary.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


class _Unit(NamedTuple):
    text: str
    n_tokens: int


def count_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    """Count tokens the way the embedding model sees them.

    add_special_tokens=False: [CLS]/[SEP] are added once per chunk at embedding
    time, not per sentence, so we don't count them here.
    verbose=False: silences the "longer than 512 tokens" warning. We are only
    counting here, and long paragraphs get split further anyway.
    """
    return len(tokenizer.encode(text, add_special_tokens=False, verbose=False))


def _hard_split(text: str, tokenizer: PreTrainedTokenizerBase, max_tokens: int) -> list[_Unit]:
    """Cut text into windows of max_tokens tokens. Last resort for huge sentences.

    We slice the ORIGINAL string using character offsets instead of decoding
    token ids, because decoding with an uncased tokenizer (like bge) would
    lowercase the text and mangle spacing.
    """
    encoding = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
    )
    offsets = encoding["offset_mapping"]
    units = []
    for start in range(0, len(offsets), max_tokens):
        window = offsets[start : start + max_tokens]
        piece = text[window[0][0] : window[-1][1]].strip()
        if piece:
            units.append(_Unit(piece, len(window)))
    return units


def split_into_units(text: str, tokenizer: PreTrainedTokenizerBase, max_tokens: int) -> list[_Unit]:
    """Split text into pieces of at most max_tokens, preferring natural boundaries."""
    units: list[_Unit] = []
    for paragraph in text.split("\n\n"):
        n = count_tokens(paragraph, tokenizer)
        if n <= max_tokens:
            units.append(_Unit(paragraph, n))
            continue
        # Paragraph too long: fall back to sentences.
        for sentence in _SENTENCE_END.split(paragraph):
            n = count_tokens(sentence, tokenizer)
            if n <= max_tokens:
                units.append(_Unit(sentence, n))
            else:
                # Sentence too long (e.g. a giant table row): hard split.
                units.extend(_hard_split(sentence, tokenizer, max_tokens))
    return [u for u in units if u.text.strip()]


def _overlap_tail(units: list[_Unit], overlap: int) -> list[_Unit]:
    """Return the last units of a chunk whose total size fits within `overlap` tokens.

    We overlap by whole units (not raw tokens) so the next chunk never starts
    mid-sentence. If the last unit alone is bigger than `overlap`, there's
    no overlap for this boundary, which is an accepted trade-off.
    """
    tail: list[_Unit] = []
    total = 0
    for unit in reversed(units):
        if total + unit.n_tokens > overlap:
            break
        tail.insert(0, unit)
        total += unit.n_tokens
    return tail


def chunk_page(
    page: Page,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    overlap: int,
    source_hash: str,
) -> list[Chunk]:
    """Split one page into chunks of at most chunk_size tokens."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not 0 <= overlap < chunk_size:
        # overlap >= chunk_size would mean a chunk is all overlap: no progress.
        raise ValueError(f"overlap must be in [0, chunk_size), got {overlap}")
    # The embedding model adds [CLS] and [SEP] to every chunk, and silently
    # truncates anything past its limit (512 for bge-small). So the usable
    # size is 2 less than the limit.
    max_size = tokenizer.model_max_length - tokenizer.num_special_tokens_to_add()
    if chunk_size > max_size:
        raise ValueError(
            f"chunk_size {chunk_size} exceeds what the embedding model can read "
            f"({max_size} tokens + special tokens); use at most {max_size}"
        )

    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    current_tokens = 0

    for unit in split_into_units(page.text, tokenizer, chunk_size):
        if current and current_tokens + unit.n_tokens > chunk_size:
            # Current chunk is full: save it and seed the next one with overlap.
            groups.append(current)
            current = _overlap_tail(current, overlap)
            current_tokens = sum(u.n_tokens for u in current)
            if current_tokens + unit.n_tokens > chunk_size:
                # Overlap + new unit still doesn't fit: drop the overlap.
                current, current_tokens = [], 0
        current.append(unit)
        current_tokens += unit.n_tokens

    if current:
        groups.append(current)

    chunks = []
    for index, group in enumerate(groups):
        text = " ".join(u.text for u in group)
        chunks.append(
            Chunk(
                # Deterministic id: same input -> same id. Needed to update or
                # delete chunks of a changed file in the vector DB later.
                chunk_id=f"{page.doc_id}-p{page.page}-c{index}",
                doc_id=page.doc_id,
                title=page.title,
                category=page.category,
                page=page.page,
                chunk_index=index,
                text=text,
                # Recount on the final text instead of trusting the sum.
                n_tokens=count_tokens(text, tokenizer),
                source_hash=source_hash,
            )
        )
    return chunks


def chunk_pages(
    pages: list[Page],
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    overlap: int,
    source_hash: str,
) -> list[Chunk]:
    """Chunk every page of one document."""
    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(chunk_page(page, tokenizer, chunk_size, overlap, source_hash))
    return chunks
