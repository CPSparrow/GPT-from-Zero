from transformers import pipeline
prompt = "尽管前向传播的算法需要在函数内部定义，但应该在调用 Module 实例之后而不是这个函数中调用，因为前者负责运行预处理和后处理步骤，而后者则默默地忽略它们"
generator = pipeline(
    "text-generation", model="./code/GPT_model/0228/checkpoint-500",
    device='cuda',
    max_new_tokens=50
)
print(generator(prompt))
