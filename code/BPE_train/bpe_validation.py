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

bpe = Tokenizer.from_file(os.path.join(bpe_dir, bpe_name))
encode = bpe.encode(text)
print("split:","|".join(bpe.decode(encode.ids)))
