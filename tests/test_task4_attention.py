import unittest

import torch

import task4


class Task4AttentionTests(unittest.TestCase):
    def test_attention_model_masks_padding_and_returns_expected_shapes(self):
        torch.manual_seed(7)
        device = torch.device("cpu")
        pad_idx = 0
        vocab_size = 13
        emb_dim = 5
        hidden_dim = 7
        batch_size = 2
        src_len = 4
        trg_len = 3

        encoder = task4.AttentionEncoder(
            vocab_size,
            emb_dim,
            hidden_dim,
            dropout=0.0,
            pad_idx=pad_idx,
            bidirectional=True,
        )
        decoder = task4.AttentionDecoder(
            vocab_size,
            emb_dim,
            hidden_dim,
            dropout=0.0,
            pad_idx=pad_idx,
        )
        model = task4.Seq2SeqAttention(encoder, decoder, pad_idx, device)

        src = torch.tensor([
            [1, 2, 3, 4],
            [5, 6, 0, 0],
        ])
        src_lens = torch.tensor([4, 2])
        trg = torch.tensor([
            [2, 8, 9],
            [2, 10, 11],
        ])

        outputs, attentions = model(src, src_lens, trg, teacher_forcing=0.0)

        self.assertEqual(outputs.shape, (batch_size, trg_len, vocab_size))
        self.assertEqual(attentions.shape, (batch_size, trg_len, src_len))
        self.assertTrue(torch.allclose(attentions[:, 1:, :].sum(dim=2), torch.ones(batch_size, trg_len - 1)))
        self.assertLess(float(attentions[1, 1:, 2:].abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
