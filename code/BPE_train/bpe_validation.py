import numpy as np
from collections import defaultdict
import jsonlines
import os
import json
from tokenizers import Tokenizer

# output be like:
# 端午节是中国的传统节日之一，粽子作为节日的象征，
# 端午|节|是|中国|的|传统|节日|之一|，|粽子|作为|节日|的|象征|，
# |端|午节|是|中国|的|传统|节日|之一|，|�|�|子|作为|节|日的|象|征|，

text = "端午节是中国的传统节日之一，粽子作为节日的象征，"
# text = 'bpe代码基本完成，随后发现训练bpe只需要大约1GB的数据即可，而非原先预想的要把所有数据跑一遍'
with open('./code/BPE_train/bpe_cfg.json', 'r', encoding='utf-8') as cfg_src:
    cfg = json.load(cfg_src)

bpe_dir = cfg['bpe_dir']
bpe_name = cfg['bpe_name']

bpe = Tokenizer.from_file("./code/BPE_tokenizer/bpe.json")
bpe_2 = Tokenizer.from_file("./code/BPE_tokenizer/bpe_50k.json")
encode = bpe.encode(text)
print("split:", "|".join(bpe.decode(encode.ids)))

res = defaultdict(list)
with open("./corpus/pretrain/基础语料2.0/01.jsonl", 'r', encoding="utf-8") as f:
    for index, item in enumerate(jsonlines.Reader(f)):
        if index >= 100:
            break
        res['30k'].append(len(bpe.encode(item['Content'])))
        res['50k'].append(len(bpe_2.encode(item['Content'])))
        res['raw'].append(len(item['Content']))

print(np.average(np.array(res['30k'])))
print(np.average(np.array(res['50k'])))
print(np.average(np.array(res['raw'])))
