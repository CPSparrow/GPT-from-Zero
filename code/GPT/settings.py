from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    A simple data class that contains model config.
    
    Attributes:
        model_pwd (str): 保存模型的文件夹路径。
        data_pwd (str): 语料库所在的文件夹路径。
        bpe_pwd (str): 分词器所在的文件夹路径。
        enable_reload (bool): 是否从模型文件夹中导入checkpoint
        max_length (int): The maximum length of input sequences.
        batch_size (int): The number of samples per batch.
        n_accumulation (int): The number of batches to accumulate.
        n_dim (int): The dimensionality of the model's embeddings.
        n_head (int): The number of attention heads.
        n_layer (int): The number of layers in the model.
        learning_rate (float): The learning rate for the optimizer.
    """
    max_length: int = 0
    batch_size: int = 0
    n_accumulation: int = 0
    n_dim: int = 0
    n_head: int = 0
    n_layer: int = 0
    learning_rate: float = 0
    model_pwd: str = ""
    data_pwd: str = ""
    bpe_pwd: str = ""
    enable_reload: bool = False
    
    def __post_init__(self):
        pwd_dict = {
            "model_pwd": self.model_pwd,
            "data_pwd": self.data_pwd,
            "bpe_pwd": self.bpe_pwd,
        }
        for name, dir in pwd_dict.items():
            assert dir is not None and dir != "", f"{name}不能为空白路径"


config = ModelConfig(
    max_length=256,
    batch_size=56,
    n_accumulation=10,
    n_dim=1024,
    n_head=16,
    n_layer=24,
    learning_rate=4e-4,
    model_pwd="",
    data_pwd="",
    bpe_pwd="",
    enable_reload=False,
)

if __name__ == '__main__':
    test_config = ModelConfig(
        max_length=256,
        batch_size=56,
        n_accumulation=10,
        n_dim=1024,
        n_head=16,
        n_layer=24,
        learning_rate=4e-4,
        model_pwd="/dir/to/save/trained/models",
        data_pwd="/dir/to/load/data/for/training",
        bpe_pwd="/dir/to/load/bpe/tokenizer",
        enable_reload=False,
    )
    print(test_config)
