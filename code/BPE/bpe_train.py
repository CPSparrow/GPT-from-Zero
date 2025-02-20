import os
import jsonlines
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    trainers,
    decoders
)


# file_path='./corpus/pretrain/基础语料2.0/01.jsonl'
# bpe_dir="./code/BPE_tokenizer"
# bpe_name='bpe.json'
# bpe_step=int(5*1e4)
# vocab_size=20_000

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def count_lines(file_path:str)->int:
    """
    返回jsonl文件的行数
    """
    return int(os.popen(f"wc -l < '{file_path}'").read().strip())


def load_jsonl(file: str, start: int, step: int):
    """
    return by yielding str

    Args:
        file (str): file path
        start (int): start index
        step (int): lines to train

    Yields:
        str: text line
    """
    with open(file, 'r', encoding='utf-8') as src:
        for index, item in enumerate(jsonlines.Reader(src)):
            if index < start:
                continue
            if index == start+step:
                break
            yield item['Content']


if __name__ == "__main__":
    import json
    with open('./code/BPE/bpe_cfg.json', 'r', encoding='utf-8') as cfg_src:
        cfg = json.load(cfg_src)

    bpe_dir = cfg['bpe_dir']
    bpe_name = cfg['bpe_name']
    bpe_step = cfg['bpe_step']
    vocab_size = cfg['vocab_size']
    # above will not change while training
    file_index = cfg['file_index']
    bpe_start = cfg['bpe_start']

    if file_index == 1 and bpe_start == 0:
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False)
    else:
        tokenizer = Tokenizer.from_file(os.path.join(bpe_dir, bpe_name))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False)

    special_tokens = ["<unk>", "<s>", "</s>"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        max_token_length=7
    )
    
    epoch=5
    for _ in range(epoch):
        file_path = "./corpus/pretrain/基础语料2.0/{:2}.jsonl".format(file_index)
        assert os.path.exists(file_path), f"路径{file_path}不存在"
        tokenizer.train_from_iterator(
            load_jsonl(file_path, bpe_start, bpe_step), trainer)
        tokenizer.decoder = decoders.ByteLevel()
        os.makedirs(bpe_dir, exist_ok=True)
        tokenizer.save(os.path.join(bpe_dir, bpe_name))
        tokenizer.model.save(bpe_dir)
