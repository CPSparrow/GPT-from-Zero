import os
from tokenizers import Tokenizer


file_path='./corpus/pretrain/基础语料2.0/01.jsonl'
bpe_dir="./code/BPE_tokenizer"
bpe_name='bpe.json'


text='端午节是中国的传统节日之一，粽子作为节日的象征，'
tokenizer = Tokenizer.from_file(os.path.join(bpe_dir,bpe_name))
id=tokenizer.encode(text).ids
print('|'.join(tokenizer.decode([i]) for i in id))