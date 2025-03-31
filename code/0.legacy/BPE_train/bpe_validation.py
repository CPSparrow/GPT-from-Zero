from transformers import AutoTokenizer

# output be like:
# 端午节是中国的传统节日之一，粽子作为节日的象征，
# 端午|节|是|中国|的|传统|节日|之一|，|粽子|作为|节日|的|象征|，
# |端|午节|是|中国|的|传统|节日|之一|，|�|�|子|作为|节|日的|象|征|，

text = [
    "端午节是中国的传统节日之一，粽子作为节日的象征，",
    "证监会官方网站近日发布关于政协十四届全国委员会第二次会议第02650号（财税金融类171号）提案答复的函。证监会在函中表示，目前我国境内设有上海、深圳、北京三家证券交易所，全面实行注册制落地后，多层次资本市场体系更加清晰，较大程度满足了不同行业、不同类型、不同成长阶段企业的需求，沪深北三家证券交易所均在成渝地区设有市场基地，为西部地区提供直接便利的资本市场服务，发挥服务实体经济高质量发展的重要作用。"
]
bpe = AutoTokenizer.from_pretrained(
    "/home/coder/Documents/CodeAndFiles/SyncFiles/code/1.llm/code/0.legacy/bpe_fast/32k_v1", max_length=4096
)
tokens = bpe(text)
for ids in tokens['input_ids']:
    print("|".join([bpe.decode(id) for id in ids]))
    print("=====")
print("v2:\n")
bpe = AutoTokenizer.from_pretrained(
    "/home/coder/Documents/CodeAndFiles/SyncFiles/code/1.llm/code/0.legacy/bpe_fast/32k_v2", max_length=4096
)
tokens = bpe(text)
for ids in tokens['input_ids']:
    print("|".join([bpe.decode(id) for id in ids]))
    print("=====")
