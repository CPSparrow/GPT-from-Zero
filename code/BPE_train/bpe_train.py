import os
import json
import jsonlines
from tqdm import tqdm
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    trainers,
    decoders
)


os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_lines(file_path: str) -> int:
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
    with open('./code/BPE_train/bpe_cfg.json', 'r', encoding='utf-8') as cfg_src:
        cfg = json.load(cfg_src)

    cfg_dir = cfg['cfg_dir']
    cfg_name = cfg['cfg_name']
    bpe_dir = cfg['bpe_dir']
    bpe_name = cfg['bpe_name']
    bpe_step = cfg['bpe_step']
    vocab_size = cfg['vocab_size']
    # above will not change while training
    file_index = cfg['file_index']
    bpe_start = cfg['bpe_start']

    if file_index == 1 and bpe_start == 0:
        bpe = Tokenizer(models.BPE())
        bpe.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False)
    else:
        bpe = Tokenizer.from_file(os.path.join(bpe_dir, bpe_name))
        bpe.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False)

    special_tokens = ["<unk>", "<s>", "</s>"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        max_token_length=7
    )

    is_interrupted = False
    num_epoch = 10
    for epoch in range(num_epoch):

        file_index = cfg['file_index']
        bpe_start = cfg['bpe_start']

        file_path = "./corpus/pretrain/基础语料2.0/{:02}.jsonl".format(file_index)
        assert os.path.exists(file_path), f"路径{file_path}不存在"
        print("===== {:2} epoch:training {:02}.jsonl from {:6} to {:6}: =====".format(
            epoch+1,file_index, bpe_start, bpe_start+bpe_step))
        try:
            bpe.train_from_iterator(
                load_jsonl(file_path, bpe_start, bpe_step), trainer
            )
            bpe.decoder = decoders.ByteLevel()
        except KeyboardInterrupt:
            is_interrupted = True
            break

        if not is_interrupted:
            max_lines = get_lines(file_path)
            print(
                f"max_lines:{max_lines} progress:{(bpe_start+bpe_step)/max_lines*100}\%")
            if bpe_start+bpe_step >= max_lines:
                cfg["file_index"] += 1
                cfg["bpe_start"] = 0
            else:
                cfg["bpe_start"] += bpe_step
            os.makedirs(bpe_dir, exist_ok=True)
            bpe.save(os.path.join(bpe_dir, bpe_name))
            bpe.model.save(bpe_dir)
            with open(os.path.join(cfg_dir, cfg_name), 'w') as file:
                json.dump(cfg, file, ensure_ascii=False, indent=4)

    if is_interrupted:
        print("training stopped")
    else:
        print(f"training ended good.")
