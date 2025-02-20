import json

data={
    "file_path":'./corpus/pretrain/基础语料2.0/01.jsonl',
    "bpe_dir":"./code/BPE_tokenizer",
    "bpe_name":'bpe.json',
    "bpe_step":int(5*1e4),
    "vocab_size":20_000
}

with open('./code/BPE/bpe_cfg.json','w') as file:
    json.dump(data,file,ensure_ascii=False,indent=4)