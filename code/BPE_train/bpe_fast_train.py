from transformers import AutoTokenizer
from bpe_core import load_data
from time import time

# original tokenizer be like:
# >> bpe.special_tokens_map
# {'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>'}

if __name__ == "__main__":
    # epoch of 7 in this code indicates of about 1GB data
    num_epoch = 7

    vocab_size = 32768
    steps = 5_0000

    is_interrupted = False
    time_total = 0

    bpe = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="gpt2",
        bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>",
        vocab_size=vocab_size, add_prefix=True
    )
    for epoch in range(num_epoch):
        time_start = time()

        print("===== epoch:{:2}/{:2} =====".format(epoch+1, num_epoch))
        try:
            bpe = bpe.train_new_from_iterator(
                load_data(epoch=epoch, steps=steps),
                vocab_size=vocab_size, length=steps, add_prefix=True
            )
        except KeyboardInterrupt:
            is_interrupted = True
            break

        time_cost = time()-time_start
        time_total += time_cost

        if not is_interrupted:
            print("training of epoch {:} cost {:.2f}s,toal:{:.2f}s".format(
                epoch+1, time_cost, time_total))
            # 保存BPE文件
            path = "./code/bpe_fast/"
            bpe.save_pretrained(save_directory=path, legacy_format=False)
        else:
            break

    if is_interrupted:
        print("training stopped by user")
