from tokenizers import Tokenizer
from datasets import load_dataset
from transformers import AutoTokenizer

bpe = Tokenizer.from_file("./code/BPE_tokenizer/bpe.json")
x = bpe.encode_batch([
    "端午节啊三回又三回。",
    "Star beat!"
    ])
print(x[1].attention_mask)

tokenizer=AutoTokenizer.from_pretrained("./code/BPE_tokenizer",add_prefix_space=True)
print(tokenizer("端午节啊三回又三回。"))

# base2=load_dataset("./corpus/pretrain/基础语料2.0",split="train[:500]")
# base2=base2.train_test_split(test_size=0.2)
# print(base2['train'][0])
