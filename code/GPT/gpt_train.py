import re
import os
from time import time
from datasets import load_dataset
from collections import defaultdict
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    BatchEncoding,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)


def token_test(tokenizer):
    en = tokenizer(
        [
            "今天是个上分的好日子啊",
            "base2=load_dataset(/corpus/split_1/基础语料2.0split=train[:500])",
            "Start by loading the first 5000 examples from the ELI5-Category dataset with the ",
            "🤗 Datasets library. This’ll give you a chance to experiment and make sure ", "everything works before spending more time training on the full dataset."
        ])
    # assert False
    for ids in en['input_ids']:
        print(tokenizer.convert_ids_to_tokens(ids))


def main():
    bpe_name = "30k"
    max_length = 1024
    batch_size = 2
    use_gpt2 = False
    split_start, split_end = 0, 20_000
    num_proc, preprocess_batch_size = os.cpu_count(), (split_end-split_start)//40

    if use_gpt2:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path="gpt2")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            f"./code/bpe_fast/{bpe_name}", max_length=max_length
        )

    # tokenizer debug
    if False:
        token_test(tokenizer)
        assert False

    def preprocess(samples):
        """
        根据map的输入处理并返回 BatchEncoding
        已经确保长度不会超过1024(应该)

        Args:
            samples (list[str]): batched strings

        Returns:
            tokens (BatchEncoding):
        """
        tokens = defaultdict(list)

        total, cnt = 0, 0
        ids, masks = list(), list()

        for passage in samples['Content']:
            ids.append(tokenizer.bos_token_id)
            masks.append(1)
            total += 1

            for sentences in re.split("\n|\r|\t|\f|。| ", passage):
                for sentence in sentences.split("。"):
                    if len(sentence) <= 5:
                        continue
                    sentence = sentence[1:] if sentence[0] == '\n' else sentence
                    sentence += "。"
                    encode = tokenizer(sentence)
                    cnt = len(encode["input_ids"])
                    if cnt > max_length:
                        # drop the too long sentence
                        continue
                    if total+cnt >= max_length-2:

                        ids.insert(0, tokenizer.bos_token_id)
                        ids.append(tokenizer.eos_token_id)

                        masks.insert(0, 1)
                        masks.append(1)

                        if len(ids) > max_length and ids[-2:] == [tokenizer.bos_token_id, tokenizer.eos_token_id]:
                            # print(f"==== tail:{tokenizer.decode(ids[-2:])}")
                            ids = ids[:-2]
                            masks = masks[:-2]

                        assert len(ids) <= max_length, \
                            f"too long add:{tokenizer.decode(ids)}\nids:{len(ids)},sentence:{len(sentence)},cnt:{cnt}"
                        tokens["input_ids"].append(ids)
                        tokens["attention_mask"].append(masks)

                        ids, masks = list(), list()
                        total = 0

                    total += cnt
                    ids.extend(encode["input_ids"])
                    masks.extend(encode["attention_mask"])

            ids.append(tokenizer.eos_token_id)
            masks.append(1)
            total += 1
        if len(ids) > 0 and len(ids) <= max_length:
            assert len(ids) <= max_length, \
                f"too long left:{tokenizer.decode(ids)}\nids:{len(ids)},sentence:{len(sentences)},cnt:{cnt}"
            tokens["input_ids"].append(ids)
            tokens["attention_mask"].append(masks)
        return BatchEncoding(tokens)

    base2 = load_dataset(
        "./corpus/基础语料2.0/split_1",
        split=f"train[{split_start}:{split_end}]", num_proc=num_proc
    )
    base2 = base2.train_test_split(test_size=0.2)
    base2 = base2.map(
        preprocess,
        batched=True,
        batch_size=preprocess_batch_size,
        num_proc=num_proc,
        remove_columns=base2["train"].column_names
    )

    # dataset validation
    if False:
        from random import randint
        for i in range(5):
            print(f"line {i+1}")
            print(tokenizer.decode(base2["train"]['input_ids'][randint(0, 800)]))
        assert False
    # note that padding strategy seems remaining to be added
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=max_length
    )

    cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_embd=1024,
        n_positions=max_length,
        n_head=16, n_layer=16,
        # attn_implementation="flash_attention_2",
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(cfg)

    model_size = sum(t.numel() for t in model.parameters())
    print("===== About to Start =====")
    print(f"Model size: {model_size/1000**2:.1f}M parameters")

    training_args = TrainingArguments(
        output_dir="./code/GPT_model",
        eval_strategy="steps",
        eval_steps=70,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        weight_decay=0.0001,
        num_train_epochs=2,
        dataloader_drop_last=True,

        # below are args used for debug or efficient training
        # use_cpu=True,
        fp16=True,
        gradient_accumulation_steps=12,
        optim="adafactor",
        torch_compile=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=base2["train"],
        eval_dataset=base2["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main_start = time()
    main()
    main_end = time()
    print(f"total time used:{main_end-main_start:.2f}s")
