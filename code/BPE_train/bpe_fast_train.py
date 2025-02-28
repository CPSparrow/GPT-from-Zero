from transformers import AutoTokenizer
from datasets import load_dataset
from time import time

# original tokenizer be like:
# >> bpe.special_tokens_map
# {'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>'}


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

    data = load_dataset("./corpus/基础语料2.0/split_1", split="train", num_proc=20)

    for i in range(int(steps*0.9)):
        yield data[epoch*20_0000+i*4]['Content']

    data = load_dataset("./corpus/c4/", split="train", num_proc=20)

    for i in range(int(steps*0.1)):
        yield data[epoch*16_000+i*3]['text']


if __name__ == "__main__":

    bpe_name = "32k_v2"

    # epoch of 7 in this code indicates of about 1GB data
    if bpe_name == "30k":
        vocab_size = 32768
        num_epoch = 7
    elif bpe_name == "65k":
        vocab_size = 65536
        num_epoch = 14
    elif bpe_name == "32k_v1":
        vocab_size = 32768
        num_epoch = 7
    elif bpe_name == "32k_v2":
        vocab_size = 32768
        num_epoch = 14
    else:
        raise ValueError("Invalid bpe size")

    steps = 5_0000

    is_interrupted = False
    time_total = 0

    bpe = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="gpt2",
        bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>",
        vocab_size=vocab_size, add_prefix=True, max_length=4096
    )

    for epoch in range(num_epoch):
        time_start = time()

        print("===== epoch:{:2}/{:2} =====".format(epoch+1, num_epoch))
        try:
            bpe = bpe.train_new_from_iterator(
                text_iterator=load_data(epoch=epoch, steps=steps),
                vocab_size=vocab_size, length=steps
            )
        except KeyboardInterrupt:
            is_interrupted = True
            break

        time_cost = time()-time_start
        time_total += time_cost

        if not is_interrupted:
            print(
                "training of epoch {:} cost {:.2f}s,toal:{:.2f}s,avg:{:.2f}".format(
                    epoch+1, time_cost, time_total, time_total/(epoch+1))
            )
            # 保存BPE文件

            path = f"./code/bpe_fast/{bpe_name}"
            bpe.save_pretrained(save_directory=path, legacy_format=False)
        else:
            break

    if is_interrupted:
        print("training stopped by user")
    else:
        print("training stopped good")
