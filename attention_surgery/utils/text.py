"""
Text processing utilities.
"""
from typing import List, Optional
import torch
from transformers import GPT2Tokenizer


class TextProcessor:
    """Handles tokenization and text processing for GPT-2."""

    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize text processor with GPT-2 tokenizer.

        Args:
            model_name: HuggingFace model name
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # Set pad token to eos token for GPT-2
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, text: str, return_tensors: str = "pt") -> torch.Tensor:
        """
        Encode text to token ids.

        Args:
            text: Input text
            return_tensors: Format ("pt" for PyTorch)

        Returns:
            Token ids tensor
        """
        return self.tokenizer.encode(text, return_tensors=return_tensors)

    def decode(self, ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """
        Decode token ids to text.

        Args:
            ids: Token ids tensor
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.squeeze().tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into token strings.

        Args:
            text: Input text

        Returns:
            List of token strings
        """
        return self.tokenizer.tokenize(text)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Convert token ids to token strings.

        Args:
            ids: List of token ids

        Returns:
            List of token strings
        """
        return self.tokenizer.convert_ids_to_tokens(ids)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.vocab_size
