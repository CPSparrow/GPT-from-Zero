from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)


def token_test(tokenizer):
    en = tokenizer(
        ["base2=load_dataset(/corpus/split_1/基础语料2.0split=train[:500])",
         "Start by loading the first 5000 examples from the ELI5-Category dataset with the ",
         "🤗 Datasets library. This’ll give you a chance to experiment and make sure ", "everything works before spending more time training on the full dataset."])
    # assert False
    de = tokenizer.convert_ids_to_tokens(en['input_ids'][3])
    id = tokenizer.bos_token_id
    print(de)


if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained(
    #     pretrained_model_name_or_path="gpt2")
    tokenizer = AutoTokenizer.from_pretrained(
        "./code/bpe_fast",  # pad_token="</s>"
    )

    if False:
        token_test(tokenizer)

    def preprocess(samples):
        texts=list()
        for line in samples['Content']:
            for sentence in line.split("。"):
                if len(sentence) > 5:
                    sentence = sentence[1:] if sentence[0] == '\n' else sentence
                    texts.append(sentence)
        return tokenizer(texts)

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # result["labels"] = result["input_ids"].copy()
        return result

    block_size = 768
    num_proc = 20
    base2 = load_dataset(
        "./corpus/基础语料2.0/split_1", split="train[:50000]", num_proc=num_proc
    )

    base2 = base2.train_test_split(test_size=0.2)
    base2 = base2.map(
        preprocess,
        batched=True,
        batch_size=1,
        num_proc=num_proc,
        remove_columns=base2["train"].column_names
    )
    base2 = base2.map(
        group_texts,
        batched=True,
        num_proc=num_proc
    )
    print(base2)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_head=12, n_layer=10,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id
    )
    model = GPT2LMHeadModel(cfg)

    steps = 500
    num_epoch=3
    batch_size=12
    training_args = TrainingArguments(
        output_dir="./code/GPT_model",
        eval_strategy="steps",
        eval_steps=steps,
        num_epoch=num_epoch,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=base2["train"],
        eval_dataset=base2["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
    print(base2)
    trainer.train()
