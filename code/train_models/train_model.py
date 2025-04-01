import os
import torch
import generate_training_data
from settings import config
from transformers import (
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)

if __name__ == "__main__":
    tokenized_oscar = generate_training_data.get_lm_data()
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=os.path.join(
            config.bpe_dir, '32k_v2_1'
        )
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    cfg = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.n_dim,
        intermediate_size=config.n_hidden,
        num_hidden_layers=config.n_layer,
        num_attention_heads=config.n_head,
        max_position_embeddings=config.max_length,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        attention_dropout=0.05,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = LlamaForCausalLM(cfg)
    
    model_size = sum(t.numel() for t in model.parameters())
    print("===== About to Start =====")
    print(f"Model size: {model_size / 1e6:.2f}M parameters")
    
    training_args = TrainingArguments(
        num_train_epochs=1.83,
        warmup_steps=1500,
        output_dir=config.model_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 3,
        learning_rate=config.learning_rate,
        gradient_accumulation_steps=config.n_accumulation,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        eval_steps=2000,
        save_steps=1000,
        logging_steps=150,
        log_level="info",
        bf16=True,
        # tf32=True,
        optim="adamw_torch_fused",
        # torch_compile=True,
        adam_beta2=0.95,
        lr_scheduler_type="cosine",
        save_total_limit=8,
        dataloader_num_workers=4,
        weight_decay=0.01,
        include_tokens_per_second=True,
        # remove_unused_columns=False,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_oscar["train"],
        eval_dataset=tokenized_oscar["test"],
        data_collator=data_collator,
    )
    trainer.train()
