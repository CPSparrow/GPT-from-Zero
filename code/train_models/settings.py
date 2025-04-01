from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    A simple data class that contains model config.

    Attributes:
        model_dir (str): 保存模型的文件夹路径。
        data_dir (str): 语料库所在的文件夹路径。
        bpe_dir (str): 分词器所在的文件夹路径。
        log_dir (str): 训练记录文件所在的路径。
        enable_reload (bool): 是否从模型文件夹中导入checkpoint
        max_length (int): The maximum length of input sequences.
        batch_size (int): The number of samples per batch.
        n_accumulation (int): The number of batches to accumulate.
        n_dim (int): The dimensionality of the model's embeddings.
        n_hidden (int): The number of hidden dims in the model.
        n_head (int): The number of attention heads.
        n_layer (int): The number of layers in the model.
        learning_rate (float): The learning rate for the optimizer.
    """
    max_length: int = 0
    batch_size: int = 0
    n_accumulation: int = 0
    n_steps: int = 0
    n_dim: int = 0
    n_hidden: int = 0
    n_head: int = 0
    n_layer: int = 0
    learning_rate: float = 0
    model_dir: str = ""
    data_dir: str = ""
    bpe_dir: str = ""
    log_dir: str = ""
    
    def __post_init__(self):
        pwd_dict = {
            "model_pwd": self.model_dir,
            "data_pwd" : self.data_dir,
            "bpe_pwd"  : self.bpe_dir,
            "log_pwd"  : self.log_dir,
        }
        for name, dir in pwd_dict.items():
            assert dir is not None and dir != "", f"{name}不能为空白路径"


config = ModelConfig(
    max_length=256,
    batch_size=128,
    n_accumulation=10,
    n_steps=10_600,
    n_dim=768,
    n_hidden=2304,
    n_head=12,
    n_layer=16,
    learning_rate=1.35e-3,
    model_dir="/home/coder/Documents/CodeAndFiles/Models/TestRun/",
    data_dir="/home/coder/Documents/CodeAndFiles/Corpus",
    bpe_dir="/home/coder/Documents/CodeAndFiles/SyncFiles/code/1.llm/code/bpe_models",
    log_dir="/home/coder/Documents/CodeAndFiles/Models/TestRun/logs",
)

if __name__ == '__main__':
    pass
    # 只在测试ModelConfig类的时候使用
    # test_config = ModelConfig(
    #     max_length=256,
    #     batch_size=56,
    #     n_accumulation=10,
    #     n_dim=1024,
    #     n_head=16,
    #     n_layer=24,
    #     learning_rate=4e-4,
    #     model_pwd="/dir/to/save/trained/models",
    #     data_pwd="/dir/to/load/data/for/training",
    #     bpe_pwd="/dir/to/load/bpe/tokenizer",
    #     log_pwd="/dir/to/save/training/logs/",
    #     enable_reload=False,
    # )
    # print(test_config)
