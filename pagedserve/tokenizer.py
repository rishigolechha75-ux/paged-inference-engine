"""Tokenizers: GPT-2 BPE (from the Hugging Face Hub) and a byte-level fallback."""

from __future__ import annotations

from typing import List, Optional, Protocol


class Tokenizer(Protocol):
    eos_token_id: Optional[int]

    def encode(self, text: str) -> List[int]: ...

    def decode(self, ids: List[int]) -> str: ...


class GPT2Tokenizer:
    """Wraps the ``tokenizers`` BPE model shipped with GPT-2 (tokenizer.json)."""

    eos_token_id: Optional[int] = 50256

    def __init__(self, repo_id: str = "openai-community/gpt2", path: Optional[str] = None) -> None:
        from tokenizers import Tokenizer as HFTokenizer

        if path is None:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id, "tokenizer.json")
        self._tok = HFTokenizer.from_file(path)

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)


class ByteTokenizer:
    """One token per UTF-8 byte (vocab 256). Used with the tiny test model."""

    eos_token_id: Optional[int] = None

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: List[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")
