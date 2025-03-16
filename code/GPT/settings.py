from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    A simple data class that contains model config.
    
    Attributes:
        model_pwd (str): 保存模型的文件夹路径。
        data_pwd (str): 语料库所在的文件夹路径。
        bpe_pwd (str): 分词器所在的文件夹路径。
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


config = ModelConfig(
    max_length=256,
    batch_size=4,
    n_accumulation=1,
    n_dim=1024,
    n_head=16,
    n_layer=24,
    learning_rate=5e-4,
    model_pwd="/home/handsome-coder/桌面/SyncFiles/code/1.llm/code/models",
    data_pwd="/home/handsome-coder/桌面/corpus/1.pretrain",
    bpe_pwd="/home/handsome-coder/桌面/SyncFiles/code/1.llm/code/bpe_fast"
)
