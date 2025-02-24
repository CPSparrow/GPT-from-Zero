from transformers import AutoTokenizer
from bpe_core import load_data

if __name__ == "__main__":
    bpe = AutoTokenizer.from_pretrained("gpt2")
    # >> bpe.special_tokens_map
    # {'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>'}

    is_interrupted = False
    time_total = 0
    # bpe=bpe.train_new_from_iterator()
