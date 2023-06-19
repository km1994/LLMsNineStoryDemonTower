# 【LLMs 入门实战】 Ziya-Visual模型学习与实战
 
- Ziya-Visual模型开源地址：https://huggingface.co/IDEA-CCNL/Ziya-BLIP2-14B-Visual-v1L/Ziya-BLIP2-14B-Visual-v1
- Demo体验地址：https://huggingface.co/spaces/IDEA-CCNL/Ziya-BLIP2-14B-Visual-v1-DemoDEA-CCNL/Ziya-BLIP2-14B-Visual-v1-Demo
- Ziya开源模型：https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1L/Ziya-LLaMA-13B-v1
- 封神榜项目主页：https://github.com/IDEA-CCNL/Fengshenbang-LM

## 一、前言

### 1.1 动机

自从3月份OpenAI发布具有识图能力的多模态大模型GPT-4，大模型的能力便不再局限于文本输入-文本输出的形式，人们可以上传视觉图片来与大模型进行聊天和交互。遗憾的是，时至今日绝大部分用户也都还没有拿到GPT-4输入图片的权限，无法体验到结合视觉和语言两大模态的大模型的卓越能力，而且GPT-4也没有叙述或者开源GPT模型多模态预训练的方案。与之相对的是，学术界和开源界则充分探索了视觉预训练模型（比如ViT, Vision Transformer）与大语言模型(LLM，Large Language Model)结合，从而让目前的LLM获得输入图片、认识图片的能力。其中的代表工作包括国外团队开源的Mini-GPT4[1]，LLaVA[2]等，国内团队开源的VisuaGLM[3]，mPLUG-Owl[4]等工作。大部分的开源方案参考了BLIP2的训练方案[5]，选择冻结LLM部分的参数训练或者采用Lora等parameter-efficient的微调训练方式。IDEA研究院封神榜团队在5月17日发布“姜子牙通用大模型v1”之后，继续发布Ziya-BLIP2-14B-Visual-v1多模态大模型（以下简称Ziya-Visual模型）。和Ziya大模型一样，Ziya-Visual模型具备中英双语能力，特别是中文能力较为突出。和所有基于BLIP2的方案类似，我们简单高效的扩展了LLM的识图能力。该模型对比VisualGLM、mPLUG-Owl模型，在视觉问答（VQA）评价和GPT-4打分评价[2]中，展现了一些优势。

### 1.2 软件资源

- CUDA 11.7
- Python 3.10
- pytorch 1.13.1+cu117

## 二、环境搭建

### 2.1 下载代码 

```s
    $ git clone 
```

### 2.2 构建环境

```s
    $ conda create -n py310_chat python=3.10       # 创建新环境
    $ source activate py310_chat                   # 激活环境
```

### 2.3 安装依赖 

```s
    $ cd pandallm
    $ pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 三、模型推理

首先加载Ziya-Visual模型：需要注意的是Visual-Ziya的模型仓库只包含视觉模型部分的参数，Ziya LLM部分的参数通过https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1L/Ziya-LLaMA-13B-v1获得。得到这两部分的模型参数后，我们加载模型：

```s
from transformers import LlamaForCausalLM, LlamaTokenizer, BlipImageProcessor
from modeling_ziya_blip2 import ZiyaBLIP2ForConditionalGeneration
from PIL import Image

# model path of IDEA-CCNL/Ziya-LLaMA-13B-v1
LM_MODEL_PATH="local path of model Ziya-LLaMA-13B-v1"
lm_model = LlamaForCausalLM.from_pretrained(LM_MODEL_PATH)
tokenizer = LlamaTokenizer.from_pretrained(LM_MODEL_PATH)

# visual model
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
# demo.py is in the project path, so we can use local path ".". Otherwise you should use "IDEA-CCNL/Ziya-BLIP2-14B-Visual-v1"
model = ZiyaBLIP2ForConditionalGeneration.from_pretrained(".", language_model=lm_model)
image_size = model.config.vision_config.image_size
image_processor = BlipImageProcessor(
    size={"height": image_size, "width": image_size},
    image_mean=OPENAI_CLIP_MEAN,
    image_std=OPENAI_CLIP_STD,
)
model.cuda() # if you use on cpu, comment this line
```

模型加载完毕后，我们就可以愉快地使用Ziya-Visual模型了：

```s
generate_config = {
    "max_new_tokens": 128,
    "top_p": 0.1,
    "temperature": 0.7
}
output = model.chat(
    tokenizer=tokenizer,
    pixel_values=image_processor(Image.open("wzry.jpg"), return_tensors="pt").pixel_values.to(model.device),
    query="这是什么游戏",
    previous_querys=[],
    previous_outputs=[],
    **generate_config,
    )
print(output)
# 这是一款名为《王者荣耀》的多人在线竞技游戏。在游戏中，玩家扮演不同的角色，并与其他玩家进行战斗。游戏中的人物和环境都是虚拟的，但它们看起来非常逼真。玩家需要使用各种技能和策略来击败对手，并获得胜利。
```

## 四、Demo部署

我们已经在huggingface上部署了Ziya-Visual的Demo，用户可以直接通过git clone获得demo的代码：

```s
git clone https://huggingface.co/spaces/IDEA-CCNL/Ziya-BLIP2-14B-Visual-v1-Demo
```

然后类似代码调用里的情况，注意将launch.py里面的LM_MODEL_PATH替换为自己本地目录下的Ziya-v1模型，之后就可以启动demo了：

```s
python launch.py
```

## 填坑笔记


## 参考

1. [Panda LLM: Training Data and Evaluation for Open-Sourced Chinese Instruction-Following Large Language Models](https://arxiv.org/pdf/2305.03025v1.pdf)
2. [dandelionsllm/pandallm](https://github.com/dandelionsllm/pandallm)
