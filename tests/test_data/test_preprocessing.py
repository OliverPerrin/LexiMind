import unittest

from LexiMind.src.data.preprocessing import TextPreprocessor
from LexiMind.src.data.tokenization import Tokenizer, TokenizerConfig


class _StubTokenizer(Tokenizer):
    def __init__(self, max_length: int) -> None:
        # Avoid expensive huggingface initialisation by skipping super().__init__
        self.config = TokenizerConfig(max_length=max_length)

    def batch_encode(self, texts, *, max_length=None):
        raise NotImplementedError


class TextPreprocessorTests(unittest.TestCase):
    def test_matching_max_length_leaves_tokenizer_unchanged(self) -> None:
        tokenizer = _StubTokenizer(max_length=128)
        TextPreprocessor(tokenizer=tokenizer, max_length=128)
        self.assertEqual(tokenizer.config.max_length, 128)

    def test_conflicting_max_length_raises_value_error(self) -> None:
        tokenizer = _StubTokenizer(max_length=256)
        with self.assertRaises(ValueError):
            TextPreprocessor(tokenizer=tokenizer, max_length=128)


if __name__ == "__main__":
    unittest.main()
