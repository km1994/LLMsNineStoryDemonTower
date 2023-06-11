# 【关于 ChatGLM + LoRA 进行finetune 】那些你不知道的事

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
