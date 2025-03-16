import os
import torch
import preprocess
import settings
from transformers import (
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)
import torch._dynamo

if __name__ == '__main__':
    torch._dynamo.config.capture_scalar_outputs = True
    
    data = preprocess.get_tokenized(None)
    
    """
    train = data["train"].to_iterable_dataset()
    cnt = 0
    for index, item in enumerate(train):
        if index >= 5000:
            break
        cnt += sum(item["attention_mask"])
    print(f"avg:{cnt / 5000:.2f}")
    """
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=os.path.join(
            settings.config.bpe_pwd, "32k_v1"
        )
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    
    cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_embd=settings.config.n_dim,
        torch_dtype=torch.bfloat16,
        n_positions=settings.config.max_length,
        n_head=settings.config.n_head,
        n_layer=settings.config.n_layer,
        attn_implementation="flash_attention_2",
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    
    model = GPT2LMHeadModel(cfg)
    model_size = sum(t.numel() for t in model.parameters())
    print("===== About to Start =====")
    print(f"Model size: {model_size / 1000 ** 2:.1f}M parameters")
    
    training_args = TrainingArguments(
        output_dir=settings.config.model_pwd,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=250,
        save_steps=500,
        per_device_train_batch_size=settings.config.batch_size,
        per_device_eval_batch_size=settings.config.batch_size,
        learning_rate=settings.config.learning_rate,
        weight_decay=0.01,
        num_train_epochs=100,
        dataloader_drop_last=True,
        bf16=True,
        gradient_accumulation_steps=settings.config.n_accumulation,
        optim="adamw_torch_fused",
        torch_compile=True,
        adam_beta2=0.98,
        lr_scheduler_type="cosine_with_restarts",
        warmup_steps=1500,
        log_level="info",
        label_smoothing_factor=0.1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["train"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    trainer.train()
    print(data)
