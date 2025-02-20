import json
import os


data = {
    "file_path": './corpus/pretrain/基础语料2.0/01.jsonl',
    "bpe_dir": "./code/BPE_tokenizer",
    "bpe_name": 'bpe.json',
    "bpe_step": int(5*1e4),
    "vocab_size": 20_000
}

# with open('./code/BPE/bpe_cfg.json','w') as file:
#     json.dump(data,file,ensure_ascii=False,indent=4)


def get_lines(file_path: str) -> int:
    """
    返回jsonl文件的行数
    """
    return int(os.popen(f"wc -l < '{file_path}'").read().strip())


print(get_lines(data['file_path']))
