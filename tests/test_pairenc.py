"""pairenc — 근거만 자르기 · 긴 주장 물러서기 (tokenizers 가 있을 때만)."""
import unittest

import helpers

HAS_TOKENIZERS = helpers.has("tokenizers")


def char_tokenizer(text, xlmr=False):
    """글자 단위 토크나이저. xlmr 이면 XLM-R 쌍 틀 `<s> A </s></s> B </s>` (특수 토큰 4), 아니면 klue 틀 (3)."""
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers, processors

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    if xlmr:
        vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
    for ch in text:
        if not ch.isspace() and ch not in vocab:
            vocab[ch] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>" if xlmr else "[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Split(Regex("."), behavior="isolated")])
    if xlmr:
        tok.post_processor = processors.RobertaProcessing(("</s>", 2), ("<s>", 0))
    else:
        tok.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    return tok


@unittest.skipUnless(HAS_TOKENIZERS, "tokenizers 없음")
class PairEncTest(unittest.TestCase):
    def setUp(self):
        from pairenc import encode_pair, pair_length

        self.encode_pair = encode_pair
        self.pair_length = pair_length
        self.tok = char_tokenizer("가나다라마바사아자차카타파하")

    def test_short_pair_not_cut(self):
        ids, cut = self.encode_pair(self.tok, "가나다", "라마", 32)
        self.assertFalse(cut)
        self.assertEqual(ids[0], 2)
        self.assertEqual(ids.count(3), 2)
        self.assertEqual(len(ids), 3 + 2 + 3)

    def test_only_evidence_is_cut_and_head_kept(self):
        evidence = "가나다라마바사아자차"
        ids, cut = self.encode_pair(self.tok, evidence, "카타", 10)
        self.assertTrue(cut)
        self.assertEqual(len(ids), 10)
        claim_ids = self.tok.encode("카타", add_special_tokens=False).ids
        self.assertEqual(ids[-3:-1], claim_ids)  # 주장은 그대로
        head = self.tok.encode("가나다라마", add_special_tokens=False).ids
        self.assertEqual(ids[1:6], head)         # 근거 앞쪽이 남는다

    def test_long_claim_falls_back(self):
        ids, cut = self.encode_pair(self.tok, "가나", "다라마바사아자차카타", 8)
        self.assertTrue(cut)
        self.assertEqual(len(ids), 8)

    def test_pair_length_ignores_previous_truncation(self):
        self.encode_pair(self.tok, "가나다라마바", "사", 6)
        self.assertEqual(self.pair_length(self.tok, "가나다라마바", "사"), 3 + 6 + 1)


@unittest.skipUnless(HAS_TOKENIZERS, "tokenizers 없음")
class XlmrPairTest(unittest.TestCase):
    """XLM-R 꼴 — 특수 토큰 넷을 토크나이저에서 센다 (예전엔 3 으로 박혀 있었다)."""

    def setUp(self):
        import pairenc

        self.p = pairenc
        self.klue = char_tokenizer("가나다라마바사아자차카타파하")
        self.tok = char_tokenizer("가나다라마바사아자차카타파하", xlmr=True)

    def test_specials_counted_from_tokenizer(self):
        self.assertEqual(self.p.pair_specials(self.klue), 3)
        self.assertEqual(self.p.pair_specials(self.tok), 4)

    def test_short_pair_layout(self):
        ids, cut = self.p.encode_pair(self.tok, "가나다", "라마", 32)
        self.assertFalse(cut)
        self.assertEqual(ids[0], 0)
        self.assertEqual(ids.count(2), 3)          # </s> 셋 : 근거 뒤 둘 · 주장 뒤 하나
        self.assertEqual(len(ids), 3 + 2 + 4)

    def test_claim_fills_all_but_specials_falls_back(self):
        # 주장 4 + 특수 4 = 8 = max_len → 근거를 0 으로 잘라도 못 맞춘다 → longest_first 로 물러서야 오류가 안 난다
        ids, cut = self.p.encode_pair(self.tok, "가나", "다라마바", 8)
        self.assertTrue(cut)
        self.assertEqual(len(ids), 8)

    def test_evidence_cut_keeps_claim(self):
        ids, cut = self.p.encode_pair(self.tok, "가나다라마바사아자차", "카타", 10)
        self.assertTrue(cut)
        self.assertEqual(len(ids), 10)
        self.assertEqual(ids[-3:-1], self.tok.encode("카타", add_special_tokens=False).ids)
        self.assertEqual(ids[1:5], self.tok.encode("가나다라", add_special_tokens=False).ids)

    def test_template_text(self):
        self.assertEqual(self.p.pair_template(self.tok), "<s> evidence </s> </s> claim </s>")
        self.assertEqual(self.p.pair_template(self.klue), "[CLS] evidence [SEP] claim [SEP]")


class TokenTypeTest(unittest.TestCase):
    def test_type_vocab_one_passes(self):
        import pairenc

        pairenc.check_no_token_types({"type_vocab_size": 1})
        pairenc.check_no_token_types({})

    def test_type_vocab_two_refused(self):
        import pairenc

        with self.assertRaises(SystemExit):
            pairenc.check_no_token_types({"type_vocab_size": 2})


if __name__ == "__main__":
    unittest.main()
