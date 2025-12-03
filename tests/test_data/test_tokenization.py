import unittest
from unittest.mock import MagicMock, patch

import torch

from src.data.tokenization import Tokenizer, TokenizerConfig


class TestTokenizer(unittest.TestCase):
    @patch("src.data.tokenization.AutoTokenizer")
    def test_tokenizer_initialization(self, mock_auto_tokenizer):
        mock_hf_tokenizer = MagicMock()
        mock_hf_tokenizer.pad_token_id = 0
        mock_hf_tokenizer.bos_token_id = 1
        mock_hf_tokenizer.eos_token_id = 2
        mock_hf_tokenizer.vocab_size = 1000
        mock_auto_tokenizer.from_pretrained.return_value = mock_hf_tokenizer

        config = TokenizerConfig(pretrained_model_name="test-model")
        tokenizer = Tokenizer(config)

        self.assertEqual(tokenizer.pad_token_id, 0)
        self.assertEqual(tokenizer.bos_token_id, 1)
        self.assertEqual(tokenizer.eos_token_id, 2)
        self.assertEqual(tokenizer.vocab_size, 1000)
        mock_auto_tokenizer.from_pretrained.assert_called_with("test-model")

    @patch("src.data.tokenization.AutoTokenizer")
    def test_encode(self, mock_auto_tokenizer):
        mock_hf_tokenizer = MagicMock()
        mock_hf_tokenizer.pad_token_id = 0
        mock_hf_tokenizer.bos_token_id = 1
        mock_hf_tokenizer.eos_token_id = 2
        mock_hf_tokenizer.encode.return_value = [10, 11, 12]
        mock_auto_tokenizer.from_pretrained.return_value = mock_hf_tokenizer

        tokenizer = Tokenizer()
        ids = tokenizer.encode("hello world")

        self.assertEqual(ids, [10, 11, 12])
        mock_hf_tokenizer.encode.assert_called()

    @patch("src.data.tokenization.AutoTokenizer")
    def test_batch_encode(self, mock_auto_tokenizer):
        mock_hf_tokenizer = MagicMock()
        mock_hf_tokenizer.pad_token_id = 0
        mock_hf_tokenizer.bos_token_id = 1
        mock_hf_tokenizer.eos_token_id = 2

        # Mock return value for __call__
        mock_hf_tokenizer.return_value = {
            "input_ids": torch.tensor([[10, 11], [12, 13]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
        }
        mock_auto_tokenizer.from_pretrained.return_value = mock_hf_tokenizer

        tokenizer = Tokenizer()
        output = tokenizer.batch_encode(["hello", "world"])

        self.assertIn("input_ids", output)
        self.assertIn("attention_mask", output)
        self.assertIsInstance(output["input_ids"], torch.Tensor)
        self.assertIsInstance(output["attention_mask"], torch.Tensor)

    @patch("src.data.tokenization.AutoTokenizer")
    def test_decode(self, mock_auto_tokenizer):
        mock_hf_tokenizer = MagicMock()
        mock_hf_tokenizer.pad_token_id = 0
        mock_hf_tokenizer.bos_token_id = 1
        mock_hf_tokenizer.eos_token_id = 2
        mock_hf_tokenizer.decode.return_value = "hello world"
        mock_auto_tokenizer.from_pretrained.return_value = mock_hf_tokenizer

        tokenizer = Tokenizer()
        text = tokenizer.decode([10, 11, 12])

        self.assertEqual(text, "hello world")
        mock_hf_tokenizer.decode.assert_called()

    @patch("src.data.tokenization.AutoTokenizer")
    def test_prepare_decoder_inputs(self, mock_auto_tokenizer):
        mock_hf_tokenizer = MagicMock()
        mock_hf_tokenizer.pad_token_id = 0
        mock_hf_tokenizer.bos_token_id = 1
        mock_hf_tokenizer.eos_token_id = 2
        mock_auto_tokenizer.from_pretrained.return_value = mock_hf_tokenizer

        tokenizer = Tokenizer()
        labels = torch.tensor([[10, 11, 2], [12, 2, 0]])  # 0 is pad

        decoder_inputs = tokenizer.prepare_decoder_inputs(labels)

        # Should shift right and prepend BOS (1)
        expected = torch.tensor([[1, 10, 11], [1, 12, 2]])

        self.assertTrue(torch.equal(decoder_inputs, expected))


if __name__ == "__main__":
    unittest.main()
