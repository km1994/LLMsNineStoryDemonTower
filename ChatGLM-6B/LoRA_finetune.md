# 【关于 ChatGLM + LoRA 进行finetune 】那些你不知道的事

## 【LLMs 入门实战系列】

### 第一层 ChatGLM-6B

1. [【ChatGLM-6B入门-一】清华大学开源中文版ChatGLM-6B模型学习与实战](ChatGLM-6B/induction.md)
   1. 介绍：ChatGLM-6B 环境配置 和 部署
2. [【ChatGLM-6B入门-二】清华大学开源中文版ChatGLM-6B模型微调实战](ChatGLM-6B/ptuning.md)
   1. ChatGLM-6B P-Tuning V2 微调：Fine-tuning the prefix encoder of the model.
3. [【ChatGLM-6B入门-三】ChatGLM 特定任务微调实战](https://articles.zsxq.com/id_3b42ukjdkwpt.html)
4. [【ChatGLM-6B入门-四】ChatGLM + LoRA 进行finetune](https://articles.zsxq.com/id_e2389qm0w0sx.html)
   1. 介绍：ChatGLM-6B LoRA 微调：Fine-tuning the low-rank adapters of the model.
5. [ChatGLM-6B 小编填坑记](https://articles.zsxq.com/id_fw7vn0mhdsnq.html)
   1. 介绍：ChatGLM-6B 在 部署和微调 过程中 会遇到很多坑，小编掉坑了很多次，为防止 后人和小编一样继续掉坑，小编索性把遇到的坑都填了。
6. [【LLMs学习】关于大模型实践的一些总结](https://articles.zsxq.com/id_il58nxrs9jxr.html)
7. [【LLMs 入门实战 —— 十一 】基于 🤗PEFT 的高效 🤖ChatGLM-6B 微调](https://articles.zsxq.com/id_7rz5jtfguuc5.html)
   1. 微调方式：
      1. ChatGLM-6B Freeze 微调：Fine-tuning the MLPs in the last n blocks of the model.
      2. ChatGLM-6B P-Tuning V2 微调：Fine-tuning the prefix encoder of the model.
      3. ChatGLM-6B LoRA 微调：Fine-tuning the low-rank adapters of the model.
8. [【LLMs 入门实战 —— 十二 】基于 本地知识库 的高效 🤖langchain-ChatGLM ](https://articles.zsxq.com/id_54vjwns5t6in.html)
   1. 介绍：langchain-ChatGLM是一个基于本地知识的问答机器人，使用者可以自由配置本地知识，用户问题的答案也是基于本地知识生成的。

### 第二层 Stanford Alpaca 7B 

- [【LLMs 入门实战 —— 五 】Stanford Alpaca 7B 模型学习与实战](https://articles.zsxq.com/id_xnt3fvp2wxz0.html)
  - 介绍：本教程提供了对LLaMA模型进行微调的廉价亲民 LLMs 学习和微调 方式，主要介绍对于 Stanford Alpaca 7B 模型在特定任务上 的 微调实验，所用的数据为OpenAI提供的GPT模型API生成质量较高的指令数据（仅52k）。

### 第三层 Chinese-LLaMA-Alpaca 

- [【LLMs 入门实战 —— 六 】Chinese-LLaMA-Alpaca 模型学习与实战](https://articles.zsxq.com/id_dqvusswrdg6c.html)
  - 介绍：本教程主要介绍了 Chinese-ChatLLaMA,提供中文对话模型 ChatLLama 、中文基础模型 LLaMA-zh 及其训练数据。 模型基于 TencentPretrain 多模态预训练框架构建

### 第四层 小羊驼 Vicuna

- [【LLMs 入门实战 —— 七 】小羊驼 Vicuna模型学习与实战](https://articles.zsxq.com/id_q9mx24q9fdab.html)
  - 介绍：UC伯克利学者联手CMU、斯坦福等，再次推出一个全新模型70亿/130亿参数的Vicuna，俗称「小羊驼」（骆马）。小羊驼号称能达到GPT-4的90%性能

### 第五层 MiniGPT-4 

- [【LLMs 入门实战 —— 八 】MiniGPT-4 模型学习与实战](https://articles.zsxq.com/id_ff0w6czthq25.html)
  - 介绍： MiniGPT-4，是来自阿卜杜拉国王科技大学的几位博士做的，它能提供类似 GPT-4 的图像理解与对话能力

### 第六层 GPT4ALL

- [【LLMs 入门实战 —— 八 】GPT4ALL 模型学习与实战](https://articles.zsxq.com/id_ff0w6czthq25.html)
  - 介绍：一个 可以在自己笔记本上面跑起来的  Nomic AI 的助手式聊天机器人，成为贫民家孩子的 福音！

### 第七层 AutoGPT

- [AutoGPT 使用和部署](https://articles.zsxq.com/id_pli0z9916126.html)
  - 介绍：Auto-GPT是一个基于ChatGPT的工具，他能帮你自动完成各种任务，比如写代码、写报告、做调研等等。使用它时，你只需要告诉他要扮演的角色和要实现的目标，然后他就会利用ChatGPT和谷歌搜索等工具，不断“思考”如何接近目标并执行，你甚至可以看到他的思考过程。

### 第八层 MOSS

- [【LLMs 入门实战 —— 十三 】MOSS 模型学习与实战](https://articles.zsxq.com/id_4vwpxod23zrc.html)
  - 介绍：MOSS是一个支持中英双语和多种插件的开源对话语言模型，moss-moon系列模型具有160亿参数，在FP16精度下可在单张A100/A800或两张3090显卡运行，在INT4/8精度下可在单张3090显卡运行。MOSS基座语言模型在约七千亿中英文以及代码单词上预训练得到，后续经过对话指令微调、插件增强学习和人类偏好训练具备多轮对话能力及使用多种插件的能力。
  - 局限性：由于模型参数量较小和自回归生成范式，MOSS仍然可能生成包含事实性错误的误导性回复或包含偏见/歧视的有害内容，请谨慎鉴别和使用MOSS生成的内容，请勿将MOSS生成的有害内容传播至互联网。若产生不良后果，由传播者自负。


## 一、前言

本教程主要介绍对于 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 模型基于 [LoRA]() 进行finetune。

以[alpaca](https://github.com/tatsu-lab/stanford_alpaca) 为例。

硬件需求

- 显卡: 显存 >= 16G (最好24G或者以上)
- 环境：
  - python>=3.8
  - cuda>=11.6, cupti, cuDNN, TensorRT等深度学习环境

## 二、环境搭建

### 2.1 构建环境

```s
    $ conda create -n py310_chat python=3.10       # 创建新环境
    $ source activate py310_chat                   # 激活环境
```

### 2.2 下载代码

```s
    $ git clone https://github.com/mymusise/ChatGLM-Tuning.git
    $ cd ChatGLM-Tuning
```

### 2.3 安装依赖

运行微调需要4.27.1版本的transformers。除 ChatGLM-6B 的依赖之外，还需要按照以下依赖

```s
    $ pip install -r requirements.txt
```

## 三、使用方法

### 3.1 训练数据下载

#### 3.1.1 数据来源

1. [alpaca](https://github.com/tatsu-lab/stanford_alpaca)

#### 3.1.2 数据介绍

本章使用 [alpaca](https://github.com/tatsu-lab/stanford_alpaca)作为本次特定任务微调实验数据。

> 样例
```s
[
    {
        "instruction": "Give three tips for staying healthy.",
        "input": "",
        "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
    },...
]
```

- 字段
  - instruction: 指令
  - input: 输入（本数据集均为空）
  - output: 输出

#### 3.1.3 转化alpaca数据集为jsonl

> 运行代码
```s
    $ python cover_alpaca2jsonl.py  --data_path data/alpaca_data.json  --save_path data/alpaca_data.jsonl 
```

> 生成数据 data/alpaca_data.jsonl 
```s
    {"text": "### Instruction:\nIdentify the odd one out.\n\n### Input:\nTwitter, Instagram, Telegram\n\n### Response:\nTelegram\nEND\n"}
    {"text": "### Instruction:\nExplain why the following fraction is equivalent to 1/4\n\n### Input:\n4/16\n\n### Response:\nThe fraction 4/16 is equivalent to 1/4 because both numerators and denominators are divisible by 4. Dividing both the top and bottom numbers by 4 yields the fraction 1/4.\nEND\n"}
    ...
```
> 注：text 中包含 Instruction、Input、Response 三个信息
> 拼接格式为  ### Instruction:\n【Instruction内容】\n\n### Input:\n【Input内容】\n\n### Response:\n【Response内容】\nEND\n

### 3.2 tokenize_dataset 下载

```s
    $ python tokenize_dataset_rows.py  --jsonl_path data/alpaca_data.jsonl  --save_path data/alpaca     --max_seq_length 128
```

> --jsonl_path 微调的数据路径, 格式jsonl, 对每行的['context']和['target']字段进行encode
> --save_path 输出路径
> --max_seq_length 样本的最大长度

### 3.3 模型 finetune

运行以下指令进行微调：

> lora 方式 finetune
```s
    $ python finetune.py \
    --dataset_path data/alpaca \
    --lora_rank 8 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --max_steps 52000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir output
    >>>
    TrainOutput(global_step=1500, training_loss=1.4622032979329427, metrics={'train_runtime': 474.9934, 'train_samples_per_second': 3.158, 'train_steps_per_second': 3.158, 'total_flos': 3781851053211648.0, 'train_loss': 1.4622032979329427, 'epoch': 3.0})
    ...
```

### 3.4 模型推理

运行以下指令进行推理：

> infer.py 文件
```s
    from modeling_chatglm import ChatGLMForConditionalGeneration
    import torch
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, device_map='auto')
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, "mymusise/chatGLM-6B-alpaca-lora")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    from cover_alpaca2jsonl import format_example

    # alpaca数据集
    instructions = [
        {'instruction': 'Give three tips for staying healthy.',
        'input': '',
        'output': '1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule.',
        },
        {'instruction': 'What are the three primary colors?',
        'input': '',
        'output': 'The three primary colors are red, blue, and yellow.',
        }
    ]

    with torch.no_grad():
        for idx, item in enumerate(instructions):
            feature = format_example(item)
            input_text = feature['context']
            ids = tokenizer.encode(input_text)
            input_ids = torch.LongTensor([ids])
            out = model.generate(
                input_ids=input_ids,
                max_length=150,
                do_sample=False,
                temperature=0
            )
            out_text = tokenizer.decode(out[0])
            answer = out_text.replace(input_text, "").replace("\nEND", "").strip()
            item['infer_answer'] = answer
            print(out_text)
            print(f"### {idx+1}.Answer:\n", item.get('output'), '\n\n')

```

> 运行 infer.py 进行 推理
```s
    $ python infer.py
    >>>
    Output exceeds the size limit. Open the full output data in a text editor
    Instruction: Give three tips for staying healthy.
    Answer: 1. Eat a balanced diet of fruits, vegetables, lean protein, and whole grains.
    2. Get regular exercise, such as walking, running, or swimming.
    3. Stay hydrated by drinking plenty of water.
    ### 1.Answer:
    1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. 
    2. Exercise regularly to keep your body active and strong. 
    3. Get enough sleep and maintain a consistent sleep schedule. 


    Instruction: What are the three primary colors?
    Answer: The three primary colors are red, blue, and yellow.
    ### 2.Answer:
    The three primary colors are red, blue, and yellow. 
```

## 参考/感谢

1. [THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
2. [mymusise/ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning) 
3. [从0到1基于ChatGLM-6B使用LaRA进行参数高效微调](https://zhuanlan.zhihu.com/p/621793987)
