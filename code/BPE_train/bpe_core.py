import os
from time import time
import json
import jsonlines
from tqdm import tqdm
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    normalizers,
    trainers,
    decoders
)

"""
这里存放着 load_data 的代码，因为经过了验证所以不需要改动也不需要调整。
这里的代码预期是训练一个 type 为 bpe 的 tokenizer
"""

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def load_data(epoch: int, steps: int):
    """
    return by yielding str

    Args:
        **epoch** (int):
        正在进行的轮数

        **step** (int): 
        要读取的行数(实际会略多一些)

    Yields:
        str: text line
    """

    raise NotImplementedError("should be rewrite with datasets")


if __name__ == "__main__":
    with open('./code/BPE_train/bpe_cfg.json', 'r', encoding='utf-8') as cfg_src:
        cfg = json.load(cfg_src)

    cfg_dir = cfg['cfg_dir']
    cfg_name = cfg['cfg_name']
    bpe_dir = cfg['bpe_dir']
    bpe_name = cfg['bpe_name']
    bpe_step = cfg['bpe_step']
    vocab_size = cfg['vocab_size']
    num_epoch = cfg['num_epoch']

    # 因为bpe；似乎足够一次训练完成，就不保留读取的代码了
    bpe = Tokenizer(models.BPE(dropout=0.1, unk_token='<unk>'))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    bpe.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.StripAccents()]
    )

    special_tokens = ["<unk>", "<s>", "</s>"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        max_token_length=6,
        min_frequency=2
    )

    is_interrupted = False
    time_total = 0
    for epoch in range(num_epoch):
        time_start = time()

        print("===== epoch:{:2}/{:2} =====".format(epoch+1, num_epoch))
        try:
            bpe.train_from_iterator(
                load_data(epoch=epoch, steps=bpe_step), trainer
            )
            bpe.decoder = decoders.ByteLevel()
        except KeyboardInterrupt:
            is_interrupted = True
            break

        time_cost = time()-time_start
        time_total += time_cost

        if not is_interrupted:
            print("training of epoch {:} cost {:.2f}s,toal:{:.2f}s".format(
                epoch+1, time_cost, time_total))
            # 保存BPE文件
            os.makedirs(bpe_dir, exist_ok=True)
            bpe.save(os.path.join(bpe_dir, bpe_name))

    if is_interrupted:
        print("===== training stopped =====")
    else:
        print(f"===== training ended good. =====")
        print("      total time       :{:.2f}s".format(time_total))
        print("average time per epoch :{:.2f}s".format(time_total/num_epoch))
