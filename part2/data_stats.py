from transformers import T5TokenizerFast

tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")                      # (1) same checkpoint as the model

def read(path):
    with open(path) as f:
        return [line.strip() for line in f]

def stats(texts):
    ids = tok(texts, add_special_tokens=False)["input_ids"]       # (2) count </s> or not?
    mean_len = sum(len(x) for x in ids) / len(ids)                                              # (3) average number of tokens per line
    vocab = len({t for x in ids for t in x})                                                 # (4) number of DISTINCT token ids across all lines
    return mean_len, vocab

for split in ["train", "dev"]:
    nl, sql = read(f"data/{split}.nl"), read(f"data/{split}.sql")
    assert len(nl) == len(sql)
    nl_len, nl_vocab = stats(nl)
    sql_len, sql_vocab = stats(sql)
    print(f"{split}: n={len(nl)}  NL len={nl_len:.2f}  SQL len={sql_len:.2f}  "
          f"NL vocab={nl_vocab}  SQL vocab={sql_vocab}")

# Look at how SentencePiece splits SQL:
print(tok.tokenize("SELECT DISTINCT flight_1.flight_id FROM flight flight_1"))                                        # (5) paste one short SQL fragment
