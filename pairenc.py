"""쌍 인코딩 한 자리 — 학습 · 평가 · 내보내기 · 서버가 모두 이 함수로 (근거, 주장) 을 토큰 열로 바꾼다.

꼴은 토크나이저의 쌍 틀을 따른다 — klue roberta `[CLS] 근거 [SEP] 주장 [SEP]` · XLM-R `<s> 근거 </s></s> 주장 </s>`.
자르기는 근거만(only_first) 하고 근거 앞쪽을 남긴다.
주장이 너무 길어 근거만 잘라서는 못 맞추면 longest_first 로 물러선다 — only_first 는 그때 오류를 낸다.
"""


def pair_specials(tokenizer):
    """쌍 꼴에 붙는 특수 토큰 수 (tokenizers.Tokenizer). klue roberta 3 · XLM-R 4."""
    return tokenizer.num_special_tokens_to_add(True)


def encode_pair(tokenizer, evidence, claim, max_len):
    """tokenizers.Tokenizer 로 쌍 하나를 인코딩한다. (ids, 잘렸나) 를 돌려준다.

    tokenizer 의 자르기 설정을 부를 때마다 바꾼다 — 한 번에 한 판만 부른다(서버는 lock, 학습은 한 스레드).
    """
    tokenizer.no_padding()
    claim_len = len(tokenizer.encode(claim, add_special_tokens=False).ids)
    strategy = "only_first"
    if claim_len + pair_specials(tokenizer) >= max_len:
        strategy = "longest_first"
    tokenizer.enable_truncation(max_len, strategy=strategy)
    enc = tokenizer.encode(evidence, claim)
    return enc.ids, len(enc.overflowing) > 0


def pair_length(tokenizer, evidence, claim):
    """자르지 않은 쌍 토큰 수 (data_prep --measure 용)."""
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return len(tokenizer.encode(evidence, claim).ids)


def pair_template(tokenizer):
    """쌍 꼴을 글로 — model.json 의 input.pair 칸. 예 `<s> evidence </s> </s> claim </s>`.

    주의 : 넘겨 받은 토크나이저의 상태를 바꾼다 — 채우기(padding)·자르기(truncation)를 끄고 되돌리지 않는다.
    """
    tokenizer.no_padding()
    tokenizer.no_truncation()
    enc = tokenizer.encode("evidence", "claim")
    parts, last = [], None
    for token, seq in zip(enc.tokens, enc.sequence_ids):
        if seq is None:
            parts.append(token)
        elif seq != last:
            parts.append(("evidence", "claim")[seq])
        last = seq
    return " ".join(parts)


def check_no_token_types(config):
    """몸통이 token_type_ids 를 쓰면 SystemExit — 이 툴은 input_ids · attention_mask 만 넣는다.

    klue roberta · XLM-R 은 type_vocab_size 1 이라 안 쓴다. config 는 transformers 설정이나 dict.
    """
    size = config.get("type_vocab_size", 1) if isinstance(config, dict) else getattr(config, "type_vocab_size", 1)
    if size and size > 1:
        raise SystemExit(f"type_vocab_size {size} — token_type_ids 를 쓰는 몸통은 이 툴 길(입력 둘)에 안 맞는다")
