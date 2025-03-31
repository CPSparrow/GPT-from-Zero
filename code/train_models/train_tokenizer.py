import os
import tqdm
import settings
import generate_training_data
from transformers import PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    processors,
    Tokenizer,
)


def get_data_for_tokenizer(dataset, current_epoch, batch_size):
    """
    yield list[str] object for tokenizer training
    
    Attributes:
        
        dataset (Dataset.dataset): dataset object returned from `extract_content`
        current_epoch (int): current epoch
        batch_size (int): batch size
    """
    yield dataset[current_epoch * batch_size:(current_epoch + 1) * batch_size]['Content']


if __name__ == "__main__":
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=32768,
        show_progress=True,
        special_tokens=["</s>"],
    )
    
    oscar = generate_training_data.extract_content()
    n_epoch = 12
    batch_size = 40_000
    for current_epoch in tqdm.tqdm(range(n_epoch)):
        print(f"epoch {current_epoch + 1} of {n_epoch}")
        tokenizer.train_from_iterator(
            get_data_for_tokenizer(oscar, current_epoch, batch_size), trainer=trainer
        )
    
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    pretrained_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="</s>",
        eos_token="</s>",
    )
    pretrained_tokenizer.save_pretrained(
        os.path.join(settings.config.bpe_dir, '32k_v2_1')
    )
    encoding = tokenizer.encode("让我们测试这个程序。")
    print(encoding.tokens)
