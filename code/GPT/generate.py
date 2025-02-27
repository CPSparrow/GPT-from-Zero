from transformers import pipeline
# model=PreTrainedModel().from_pretrained("./code/GPT_model/checkpoint-648")
prompt = "村民们都很高兴，"
generator = pipeline(
    "text-generation", model="./code/GPT_model/checkpoint-648", device='cuda',
    max_new_tokens=50
)
print(generator(prompt))
