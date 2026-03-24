# pipeline/process/chunk.py
# Split plain text into overlapping chunks ready for embedding.
#
# Token approximation: one token ≈ one whitespace-delimited word.
# Target size is 500 tokens; chunks outside [400, 600] are only produced when
# the source text is shorter than the minimum or longer than one chunk.
# Overlap of 50 tokens carries context across chunk boundaries.

from dataclasses import dataclass

CHUNK_SIZE = 500      # target words per chunk
CHUNK_OVERLAP = 50   # words carried from previous chunk
MIN_TOKENS = 100     # pages with fewer tokens are discarded as stubs


@dataclass(frozen=True)
class Chunk:
    source: str       # 'zelda_wiki', 'zelda_dungeon', or 'compendium/<category>'
    page_title: str
    chunk_index: int
    text: str

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "page_title": self.page_title,
            "chunk_index": self.chunk_index,
            "text": self.text,
        }


def _words(text: str) -> list[str]:
    return text.split()


def chunk_text(
    text: str,
    source: str,
    page_title: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_tokens: int = MIN_TOKENS,
) -> list[Chunk]:
    """Split *text* into overlapping word-based chunks.

    Returns an empty list if the token count is below *min_tokens* (stub page).
    """
    words = _words(text)
    if len(words) < min_tokens:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = chunk_size - overlap

    while start < len(words):
        end = start + chunk_size
        slice_words = words[start:end]
        chunk_text_str = " ".join(slice_words)
        chunks.append(Chunk(source=source, page_title=page_title, chunk_index=index, text=chunk_text_str))
        index += 1
        if end >= len(words):
            break
        start += step

    return chunks
