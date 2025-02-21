import transformers
model=transformers.GPT2LMHeadModel.from_pretrained('gpt-2')
print(model.parameters())