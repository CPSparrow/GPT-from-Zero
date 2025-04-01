import os
import glob
import openai
import torch
from transformers import AutoTokenizer, AutoModel, pipeline

bpe_dir = "/home/coder/Documents/CodeAndFiles/SyncFiles/code/1.llm/code/bpe_models"
model_dir = "/home/coder/Documents/CodeAndFiles/Models/TinyLLaMa"

text = [
    "尊敬的业主：因电缆检修，本小区将于本周六8:00-12:00暂停供水。请提前做好储水准备。另，关于流浪猫投喂引发的纠纷...",
    "尊敬的顾客，今日生鲜区推出限时折扣：山东红富士苹果3.98元/斤，越南白心火龙果买二送一。温馨提示：水产柜台的王师傅今天值夜班...",
    "现在插播路况：建国路南向北方向因水管爆裂占用车道，建议绕行朝阳路。接下来是整点新闻：市统计局最新数据显示，本季度二手房成交量...",
    "智能电饭煲使用指南：请勿将内胆直接置于明火加热。清洁时避免使用钢丝球。常见问题：若显示屏出现E-3代码，代表...",
    "工作经历：2019.3-2022.8 任XX公司行政专员，负责办公用品采购、会议记录及考勤管理。离职原因：因家庭原因需迁往外地...",
    "记一次运动会：上周五我校举办了春季运动会。我参加了60米短跑比赛，虽然只得了第三名，但王老师说...",
    "姓名：张三 诊断：过敏性鼻炎 R：氯雷他定片 10mg×6片 用法：每日一次，每次一片。注意事项：服药期间避免食用...",
    "2023年4月15日物业协调会决议：1.同意将东门快递柜移至3号楼北侧 2.关于停车费调整方案需进一步征集业主意见...",
    "今日夜间晴转多云，西北风3-4级，气温18℃~26℃。明日午后有分散性雷阵雨，请市民注意携带雨具。空气质量指数：良...",
    "诚聘收银员：要求年龄18-35岁，高中以上学历，能适应轮班。薪资待遇：底薪3500+绩效提成。面试需携带身份证原件及...",
    "@全体成员 今日数学作业：练习册P25-27页。另，下周五将进行期中考试，请督促孩子复习。温馨提示：近期诺如病毒高发...",
    "本产品享受全国联保，整机保修一年，主要部件保修三年。下列情况不在保修范围：1.人为损坏 2.未保留原始购买凭证...",
    "【XX银行】您尾号3579的信用卡于4月17日14:23消费人民币286元，商户为XX超市。本期账单应还金额...",
    "第七条：租赁期间产生的水费、电费、物业费由乙方承担。退租时需结清所有费用并保证房屋设施完好，如有损坏照价赔偿...",
    "本周团体课程安排：周一19:00 瑜伽初级班 周二20:00 搏击操 周三休息 周四18:30 动感单车 请会员提前10分钟签到...",
    "宠物姓名：雪球 品种：英国短毛猫 诊疗记录：4月10日注射三联疫苗，4月15日复查时发现耳螨，建议...",
    "番茄炒蛋：1.鸡蛋打散加盐搅拌 2.热锅凉油炒至定型盛出 3.另起锅炒番茄至出汁 4.混合翻炒加糖调味。小贴士：...",
    "收费标准：小型车白天时段（8:00-20:00）6元/小时，夜间时段（20:00-8:00）3元/小时。月租车请至管理处办理...",
    "本研究通过问卷调查法，对500名大学生进行社交媒体使用与睡眠质量相关性分析。结果显示，日均使用时长超过3小时的群体...",
    "报修时间：2023-4-17 15:30 故障描述：空调制冷效果减弱，运行时有异响。维修人员回复：需更换压缩机，预计费用...",
]


def get_path():
    checkpoint_dir = os.path.join(model_dir, "checkpoint-*")
    checkpoints = [
        path for path in glob.glob(checkpoint_dir) if os.path.isdir(path)
    ]
    checkpoints = sorted(checkpoints, key=lambda x: os.path.getmtime(x), reverse=True)
    
    if True:
        print("===== checkpoints =====", checkpoints, sep='\n')
    return checkpoints


def generate_from_file(prompt, length, tokenizer, model_index=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_index is None:
        model_index = 0
    model = get_path()[model_index]
    pipe = pipeline(
        task="text-generation", device=device, model=model,
        max_new_tokens=length, tokenizer=tokenizer,
    )
    pipe.tokenizer.pad_token_id = tokenizer.eos_token_id
    ans = pipe(prompt, batch_size=10)
    print("model " + model.split('/')[-1] + ":\n")
    res = list()
    for x, y in zip(ans, prompt):
        t = x[0]['generated_text']
        res.append({'prompt': y, 'ans': t.split(y)[-1]})
    return res


def generate_from_server(prompt, length):
    """
    使用本地部署的 OpenAI 服务生成文本。

    Args:
        prompt (str): 输入提示文本。
        length (int): 生成文本的最大长度（token 数量）。

    Returns:
        str: 模型生成的文本。
    """
    client = openai.OpenAI(
        base_url="http://127.0.0.1:1234/v1/",
        api_key="not-needed"
    )
    response = client.chat.completions.create(
        model="qwen2.5-1.5b-instruct",  # 替换为实际的模型名称
        messages=[
            {'role': 'system', 'content': "续写以下内容："},
            {'role': 'user', 'content': f"{prompt}"},
        ],
        temperature=0.3,
        max_tokens=length,
    )
    
    generated_text = response.choices[0].message
    return {'prompt': prompt, 'ans': generated_text.content}


def generate(prompt, length, style='file', model_index=None):
    """
    file for local
    server for lmstudio
    """
    if style == 'file':
        assert isinstance(prompt, list), "prompt must be a list"
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(bpe_dir, "32k_v2_1"), padding_side="left"
        )
        return generate_from_file(prompt, length, tokenizer, model_index)
    
    elif style == 'server':
        assert isinstance(prompt, list), "prompt must be a list"
        response = list()
        for i in prompt:
            response.append(generate_from_server(i, length))
        return response


if __name__ == "__main__":
    res = generate(text, length=50, style='file')
    for r in res:
        print(r)
    # res = generate(text, length=100, style='server')
    # for r in res:
    #     print(r)
