import os
import settings
from transformers import pipeline

print(settings)
context = "尽管前向传播的算法需要在函数内部定义，但应该在调用 Module 实例之后而不是这个函数中调用，因为前者负责运行预处理和后处理步骤，而后者则默默地忽略它们。"
model_dir = "/home/handsome-coder/桌面/trained_models/gpt_v2"
generator = pipeline(
    "text-generation", model=os.path.join(model_dir, "checkpoint-1600"),
    device='cpu',
    max_new_tokens=50
)
print(generator(context))
