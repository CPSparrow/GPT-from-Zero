import os
import glob
import preprocess
from code.train_models import settings
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
    
    data = preprocess.get_tokenized()
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=os.path.join(
            settings.config.bpe_dir, "32k_v1"
        )
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    
    cfg = GPT2Config(
        n_embd=settings.config.n_dim,
        n_positions=settings.config.max_length,
        n_head=settings.config.n_head,
        n_layer=settings.config.n_layer,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    model = GPT2LMHeadModel(cfg)
    model_size = sum(t.numel() for t in model.parameters())
    print("===== About to Start =====")
    print(f"Model size: {model_size / 1000 ** 2:.2f}M parameters")
    
    training_args = TrainingArguments(
        output_dir=settings.config.model_dir,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        eval_steps=2000,
        save_steps=1000,
        logging_steps=300,
        per_device_train_batch_size=settings.config.batch_size,
        per_device_eval_batch_size=settings.config.batch_size,
        learning_rate=settings.config.learning_rate,
        weight_decay=0.01,
        num_train_epochs=2,
        bf16=True,
        gradient_accumulation_steps=settings.config.n_accumulation,
        optim="adamw_torch_fused",
        torch_compile=True,
        adam_beta2=0.98,
        lr_scheduler_type="cosine_with_restarts",
        warmup_steps=1500,
        log_level="info",
        save_total_limit=6,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    if settings.config.enable_reload:
        # 检查是否存在之前的检查点
        checkpoint_dir = os.path.join(training_args.output_dir, "checkpoint-*")
        checkpoints = [
            path for path in glob.glob(checkpoint_dir) if os.path.isdir(path)
        ]
        latest_checkpoint = max(checkpoints, key=os.path.getmtime) \
            if checkpoints else None
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        data_collator=data_collator,
    )
    if settings.config.enable_reload:
        assert latest_checkpoint is not None, "检查点导入有误"
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        trainer.train()
