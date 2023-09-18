## 【LLMs 入门实战系列】

<img src="img/微信截图_20230918094559.png" width="50%" >

> 【LLMs 入门实战系列】交流群 (注：人满 可 添加 小编wx：yzyykm666 加群！)

<img src="img/微信截图_20210301212242.png" width="50%" >

- [【LLMs 入门实战系列】](#llms-入门实战系列)
  - [第一层 LLMs to Natural Language Processing (NLP)](#第一层-llms-to-natural-language-processing-nlp)
    - [第一重 ChatGLM-6B 系列](#第一重-chatglm-6b-系列)
      - [ChatGLM-6B](#chatglm-6b)
      - [ChatGLM2-6B](#chatglm2-6b)
    - [第十一重 Llama2](#第十一重-llama2)
    - [第十重 Baichuan](#第十重-baichuan)
      - [Baichuan-13B](#baichuan-13b)
      - [baichuan-7B](#baichuan-7b)
    - [第二重 Stanford Alpaca 7B](#第二重-stanford-alpaca-7b)
    - [第三重 Chinese-LLaMA-Alpaca](#第三重-chinese-llama-alpaca)
    - [第四重 小羊驼 Vicuna](#第四重-小羊驼-vicuna)
    - [第五重 MOSS](#第五重-moss)
    - [第六重 BLOOMz](#第六重-bloomz)
    - [第七重 BELLE](#第七重-belle)
    - [第八重 ChatRWKV](#第八重-chatrwkv)
    - [第九重 ChatGPT](#第九重-chatgpt)
  - [第九层 LLMs to interview](#第九层-llms-to-interview)
  - [第八层 LLMs to Inference acceleration](#第八层-llms-to-inference-acceleration)
  - [第二层 LLMs to Parameter Efficient Fine-Tuning (PEFT)](#第二层-llms-to-parameter-efficient-fine-tuning-peft)
    - [第一重 分布式训练神器](#第一重-分布式训练神器)
    - [第二重 LLMs Trick](#第二重-llms-trick)
    - [第三重 LLMTune](#第三重-llmtune)
    - [第四重 QLoRA](#第四重-qlora)
  - [第三层 LLMs to Artifact](#第三层-llms-to-artifact)
    - [第一重 langchain](#第一重-langchain)
    - [第二重 wenda](#第二重-wenda)
    - [第三重 AutoGPT](#第三重-autogpt)
    - [第四重 Knowledge Extraction](#第四重-knowledge-extraction)
  - [第四层 LLMs to Text-to-Image](#第四层-llms-to-text-to-image)
    - [第一重 Stable Diffusion](#第一重-stable-diffusion)
  - [第五层 LLMs to Visual Question Answering (VQA)](#第五层-llms-to-visual-question-answering-vqa)
    - [第一重 BLIP](#第一重-blip)
    - [第二重 BLIP2](#第二重-blip2)
    - [第三重 MiniGPT-4](#第三重-minigpt-4)
    - [第四重 VisualGLM-6B](#第四重-visualglm-6b)
    - [第五重 Ziya-Visual](#第五重-ziya-visual)
  - [第六层 LLMs to Automatic Speech Recognition (ASR)](#第六层-llms-to-automatic-speech-recognition-asr)
    - [第一重 Massively Multilingual Speech (MMS，大规模多语种语音)](#第一重-massively-multilingual-speech-mms大规模多语种语音)
    - [第二重 whisper](#第二重-whisper)
  - [第七层 LLMs to Text To Speech (TTS)](#第七层-llms-to-text-to-speech-tts)
    - [第一重 Massively Multilingual Speech (MMS，大规模多语种语音)](#第一重-massively-multilingual-speech-mms大规模多语种语音-1)
    - [第二重 Retrieval-based-Voice-Conversion](#第二重-retrieval-based-voice-conversion)
  - [LLaMA 衍生物系列](#llama-衍生物系列)
    - [第五重 GPT4ALL](#第五重-gpt4all)
    - [第十一重 OpenBuddy](#第十一重-openbuddy)
    - [第十二重 Baize](#第十二重-baize)
    - [第十三重 OpenChineseLLaMA](#第十三重-openchinesellama)
    - [第十四重 Panda](#第十四重-panda)
    - [第十五重 Ziya-LLaMA-13B](#第十五重-ziya-llama-13b)
    - [第十六重 BiLLa](#第十六重-billa)
    - [第十七重 Luotuo-Chinese-LLM](#第十七重-luotuo-chinese-llm)
    - [第十八重 Linly](#第十八重-linly)
    - [第十九重 ChatYuan](#第十九重-chatyuan)
    - [第二十重 CPM-Bee](#第二十重-cpm-bee)
    - [第二十一重 TigerBot](#第二十一重-tigerbot)
    - [第二十二重 书生·浦语](#第二十二重-书生浦语)
    - [第二十三重 Aquila](#第二十三重-aquila)
    - [第一重 金融领域](#第一重-金融领域)
    - [第二重 医疗领域](#第二重-医疗领域)
    - [第三重 法律领域](#第三重-法律领域)
    - [第四重 教育领域](#第四重-教育领域)
    - [第五重 文化领域](#第五重-文化领域)
    - [第六重 Coding](#第六重-coding)
- [知识体系](#知识体系)
- [加入学习群](#加入学习群)
- [参考](#参考)


### 第一层 LLMs to Natural Language Processing (NLP)

#### 第一重 ChatGLM-6B 系列

##### ChatGLM-6B 

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
8. [【LLMs 入门实战】基于 🤗QLoRA 的高效 🤖ChatGLM-6B 微调](https://articles.zsxq.com/id_jwxc99b1gbfd.html)
   1. 介绍：本项目使用 https://github.com/huggingface/peft  库，实现了 ChatGLM-6B 模型4bit的 QLoRA 高效微调，可以在一张RTX3060上完成全部微调过程。
9. [【LLMs 入门实战 】🤖ChatGLM-6B 模型结构代码解析](https://articles.zsxq.com/id_vi2qatrhvzwt.html)
   1.  介绍：ChatGLM-6B 模型结构代码解析

##### ChatGLM2-6B 

- [【ChatGLM2-6B入门】清华大学开源中文版ChatGLM-6B模型学习与实战](https://mp.weixin.qq.com/s/11jCCeOpg1YbABIRLlnyvg)
  - 论文名称：ChatGLM2-6B: An Open Bilingual Chat LLM | 开源双语对话语言模型
  - 论文地址：
  - Github 代码：https://github.com/THUDM/ChatGLM2-6B
  - 动机：在主要评估LLM模型中文能力的 C-Eval 榜单中，截至6月25日 ChatGLM2 模型以 71.1 的分数位居 Rank 0 ，ChatGLM2-6B 模型以 51.7 的分数位居 Rank 6，是榜单上排名最高的开源模型。
  - 介绍：ChatGLM2-6B 是开源中英双语对话模型 ChatGLM-6B 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，ChatGLM2-6B 引入了如下新特性：
    - **更强大的性能**：基于 ChatGLM 初代模型的开发经验，我们全面升级了 ChatGLM2-6B 的基座模型。ChatGLM2-6B 使用了 GLM 的混合目标函数，经过了 1.4T 中英标识符的预训练与人类偏好对齐训练，评测结果显示，相比于初代模型，ChatGLM2-6B 在 MMLU（+23%）、CEval（+33%）、GSM8K（+571%） 、BBH（+60%）等数据集上的性能取得了大幅度的提升，在同尺寸开源模型中具有较强的竞争力。
    - **更长的上下文**：基于 FlashAttention 技术，我们将基座模型的上下文长度（Context Length）由 ChatGLM-6B 的 2K 扩展到了 32K，并在对话阶段使用 8K 的上下文长度训练，允许更多轮次的对话。但当前版本的 ChatGLM2-6B 对单轮超长文档的理解能力有限，我们会在后续迭代升级中着重进行优化。
    - **更高效的推理**：基于 Multi-Query Attention 技术，ChatGLM2-6B 有更高效的推理速度和更低的显存占用：在官方的模型实现下，推理速度相比初代提升了 42%，INT4 量化下，6G 显存支持的对话长度由 1K 提升到了 8K。
    - **更开放的协议**：ChatGLM2-6B 权重对学术研究完全开放，在获得官方的书面许可后，亦允许商业使用。如果您发现我们的开源模型对您的业务有用，我们欢迎您对下一代模型 ChatGLM3 研发的捐赠。

- [【关于 ChatGLM2 + LoRA 进行finetune 】那些你不知道的事](https://articles.zsxq.com/id_pmijwhnwxwen.html)
  - 论文名称：ChatGLM2-6B: An Open Bilingual Chat LLM | 开源双语对话语言模型
  - 论文地址：
  - Github 代码：https://github.com/THUDM/ChatGLM2-6B
  - 介绍：本教程主要介绍对于 ChatGLM2-6B 模型基于 LoRA 进行finetune。

- [【LLMs 入门实战 】基于 🤗PEFT 的高效 🤖ChatGLM2-6B 微调](https://articles.zsxq.com/id_q04evzros1ca.html)
  - 微调方式：
    - ChatGLM2-6B Freeze 微调：Fine-tuning the MLPs in the last n blocks of the model.
    - ChatGLM2-6B P-Tuning V2 微调：Fine-tuning the prefix encoder of the model.
    -  ChatGLM2-6B LoRA 微调：Fine-tuning the low-rank adapters of the model.
- [【LLMs 入门实战】基于 🤗QLoRA 的高效 🤖ChatGLM2-6B 微调](https://articles.zsxq.com/id_5oocf5c9e4te.html)
  - 介绍：本项目使用 https://github.com/huggingface/peft  库，实现了 ChatGLM2-6B 模型4bit的 QLoRA 高效微调，可以在一张RTX3060上完成全部微调过程。

#### 第十一重 Llama2

- [【LLMs 入门实战】 Llama2 模型学习与实战](https://articles.zsxq.com/id_ivi7gtxdpq4p.html)
  - 官网：https://ai.meta.com/llama/
  - 论文名称：《Llama 2: Open Foundation and Fine-Tuned Chat Models》
  - 论文地址：https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/
  - 演示平台：https://llama2.ai/
  - Github 代码：https://github.com/facebookresearch/llama
  - 模型下载地址：https://ai.meta.com/resources/models-and-libraries/llama-downloads/
  - 介绍：此次 Meta 发布的 Llama 2 模型系列包含 70 亿、130 亿和 700 亿三种参数变体。此外还训练了 340 亿参数变体，但并没有发布，只在技术报告中提到了。据介绍，相比于 Llama 1，Llama 2 的训练数据多了 40%，上下文长度也翻倍，并采用了分组查询注意力机制。具体来说，Llama 2 预训练模型是在 2 万亿的 token 上训练的，精调 Chat 模型是在 100 万人类标记数据上训练的。

- [【LLMs 入门实战】Chinese-Llama-2-7b 模型学习与实战](https://articles.zsxq.com/id_gxfww20xkje7.html)
  - 官网：https://ai.meta.com/llama/
  - 论文名称：《Llama 2: Open Foundation and Fine-Tuned Chat Models》
  - 论文地址：https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/
  - 演示平台：https://huggingface.co/spaces/LinkSoul/Chinese-Llama-2-7b
  - Github 代码：https://github.com/LinkSoul-AI/Chinese-Llama-2-7b
  - 模型下载地址：
    - https://huggingface.co/ziqingyang/chinese-llama-2-7b
    - https://huggingface.co/LinkSoul/Chinese-Llama-2-7b-4bit
  - 介绍：自打 LLama-2 发布后就一直在等大佬们发布 LLama-2 的适配中文版，也是这几天蹲到了一版由 LinkSoul 发布的 Chinese-Llama-2-7b，其共发布了一个常规版本和一个 4-bit 的量化版本，今天我们主要体验下 Llama-2 的中文逻辑顺便看下其训练样本的样式，后续有机会把训练和微调跑起来。

#### 第十重 Baichuan

##### Baichuan-13B

- [【LLMs 入门实战 】 Baichuan-13B 模型学习与实战](https://articles.zsxq.com/id_a88op9no2xwi.html)
  - baichuan-inc/Baichuan-13B：https://github.com/baichuan-inc/Baichuan-13B
  - Baichuan-13B 大模型：
    - 官方微调过（指令对齐）:https://huggingface.co/baichuan-inc/Baichuan-13B-Chat
    - 预训练大模型（未经过微调）:https://huggingface.co/baichuan-inc/Baichuan-13B-Base
  - 介绍：Baichuan-13B 是由百川智能继 Baichuan-7B 之后开发的包含 130 亿参数的开源可商用的大规模语言模型，在权威的中文和英文 benchmark 上均取得同尺寸最好的效果。Baichuan-13B 有如下几个特点：
    - 更大尺寸、更多数据：Baichuan-13B 在 Baichuan-7B 的基础上进一步扩大参数量到 130 亿，并且在高质量的语料上训练了 1.4 万亿 tokens，超过 LLaMA-13B 40%，是当前开源 13B 尺寸下训练数据量最多的模型。支持中英双语，使用 ALiBi 位置编码，上下文窗口长度为 4096。
    - 同时开源预训练和对齐模型：预训练模型是适用开发者的“基座”，而广大普通用户对有对话功能的对齐模型具有更强的需求。因此本次开源同时发布了对齐模型（Baichuan-13B-Chat），具有很强的对话能力，开箱即用，几行代码即可简单的部署。
    - 更高效的推理：为了支持更广大用户的使用，本次同时开源了 int8 和 int4 的量化版本，相对非量化版本在几乎没有效果损失的情况下大大降低了部署的机器资源门槛，可以部署在如 Nvidia 3090 这样的消费级显卡上。
    - 开源免费可商用：Baichuan-13B 不仅对学术研究完全开放，开发者也仅需邮件申请并获得官方商用许可后，即可以免费商用。

##### baichuan-7B

- [【LLMs 入门实战 】 baichuan-7B 学习与实战]()
  - 论文名称：
  - 论文地址：
  - Github 代码： https://github.com/baichuan-inc/baichuan-7B
  - 模型：
  - 介绍：由百川智能开发的一个开源可商用的大规模预训练语言模型。基于Transformer结构，在大约1.2万亿tokens上训练的70亿参数模型，支持中英双语，上下文窗口长度为4096。在标准的中文和英文权威benchmark（C-EVAL/MMLU）上均取得同尺寸最好的效果。

#### 第二重 Stanford Alpaca 7B 

- [【LLMs 入门实战 —— 五 】Stanford Alpaca 7B 模型学习与实战](https://articles.zsxq.com/id_xnt3fvp2wxz0.html)
  - 介绍：本教程提供了对LLaMA模型进行微调的廉价亲民 LLMs 学习和微调 方式，主要介绍对于 Stanford Alpaca 7B 模型在特定任务上 的 微调实验，所用的数据为OpenAI提供的GPT模型API生成质量较高的指令数据（仅52k）。

#### 第三重 Chinese-LLaMA-Alpaca 

- [【LLMs 入门实战 —— 六 】Chinese-LLaMA-Alpaca 模型学习与实战](https://articles.zsxq.com/id_dqvusswrdg6c.html)
  - 介绍：本教程主要介绍了 Chinese-ChatLLaMA,提供中文对话模型 ChatLLama 、中文基础模型 LLaMA-zh 及其训练数据。 模型基于 TencentPretrain 多模态预训练框架构建

#### 第四重 小羊驼 Vicuna

- [【LLMs 入门实战 —— 七 】小羊驼 Vicuna模型学习与实战](https://articles.zsxq.com/id_q9mx24q9fdab.html)
  - 介绍：UC伯克利学者联手CMU、斯坦福等，再次推出一个全新模型70亿/130亿参数的Vicuna，俗称「小羊驼」（骆马）。小羊驼号称能达到GPT-4的90%性能

#### 第五重 MOSS

- [【LLMs 入门实战 —— 十三 】MOSS 模型学习与实战](https://articles.zsxq.com/id_4vwpxod23zrc.html)
  - 介绍：MOSS是一个支持中英双语和多种插件的开源对话语言模型，moss-moon系列模型具有160亿参数，在FP16精度下可在单张A100/A800或两张3090显卡运行，在INT4/8精度下可在单张3090显卡运行。MOSS基座语言模型在约七千亿中英文以及代码单词上预训练得到，后续经过对话指令微调、插件增强学习和人类偏好训练具备多轮对话能力及使用多种插件的能力。
  - 局限性：由于模型参数量较小和自回归生成范式，MOSS仍然可能生成包含事实性错误的误导性回复或包含偏见/歧视的有害内容，请谨慎鉴别和使用MOSS生成的内容，请勿将MOSS生成的有害内容传播至互联网。若产生不良后果，由传播者自负。

#### 第六重 BLOOMz

- [【LLMs 入门实战 —— 十四 】 BLOOMz 模型学习与实战](https://articles.zsxq.com/id_wd97899pkjqj.html)
  - 介绍：大型语言模型（LLMs）已被证明能够根据一些演示或自然语言指令执行新的任务。虽然这些能力已经导致了广泛的采用，但大多数LLM是由资源丰富的组织开发的，而且经常不对公众开放。作为使这一强大技术民主化的一步，我们提出了BLOOM，一个176B参数的开放性语言模型，它的设计和建立要感谢数百名研究人员的合作。BLOOM是一个仅有解码器的Transformer语言模型，它是在ROOTS语料库上训练出来的，该数据集包括46种自然语言和13种编程语言（共59种）的数百个来源。我们发现，BLOOM在各种基准上取得了有竞争力的性能，在经历了多任务提示的微调后，其结果更加强大。
  - 模型地址：https://huggingface.co/bigscience/bloomz

#### 第七重 BELLE

- [【LLMs 入门实战 —— 十五 】 BELLE 模型学习与实战](https://articles.zsxq.com/id_gxebzsadfpr2.html)
  - 介绍：相比如何做好大语言模型的预训练，BELLE更关注如何在开源预训练大语言模型的基础上，帮助每一个人都能够得到一个属于自己的、效果尽可能好的具有指令表现能力的语言模型，降低大语言模型、特别是中文大语言模型的研究和应用门槛。为此，BELLE项目会持续开放指令训练数据、相关模型、训练代码、应用场景等，也会持续评估不同训练数据、训练算法等对模型表现的影响。BELLE针对中文做了优化，模型调优仅使用由ChatGPT生产的数据（不包含任何其他数据）。
  - github 地址: https://github.com/LianjiaTech/BELLE

#### 第八重 ChatRWKV

- [【LLMs 入门实战 —— 十八 】 ChatRWKV 模型学习与实战](https://articles.zsxq.com/id_dw7jhxq736bw.html)
  - 目前 RWKV 有大量模型，对应各种场景，各种语言，请选择合适的模型：
    - Raven 模型：适合直接聊天，适合 +i 指令。有很多种语言的版本，看清楚用哪个。适合聊天、完成任务、写代码。可以作为任务去写文稿、大纲、故事、诗歌等等，但文笔不如 testNovel 系列模型。
    - Novel-ChnEng 模型：中英文小说模型，可以用 +gen 生成世界设定（如果会写 prompt，可以控制下文剧情和人物），可以写科幻奇幻。不适合聊天，不适合 +i 指令。
    - Novel-Chn 模型：纯中文网文模型，只能用 +gen 续写网文（不能生成世界设定等等），但是写网文写得更好（也更小白文，适合写男频女频）。不适合聊天，不适合 +i 指令。
    - Novel-ChnEng-ChnPro 模型：将 Novel-ChnEng 在高质量作品微调（名著，科幻，奇幻，古典，翻译，等等）。
  - github: https://github.com/BlinkDL/ChatRWKV
  -  模型文件：https://huggingface.co/BlinkDL

#### 第九重 ChatGPT

- [《ChatGPT Prompt Engineering for Developers》 学习 之 如何 编写 Prompt？](https://articles.zsxq.com/id_2wutbr4eehzm.html)
  - [吴恩达老师与OpenAI合作推出《ChatGPT Prompt Engineering for Developers》](https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/1/introduction)
  - 动机：吴恩达老师与OpenAI合作推出《ChatGPT Prompt Engineering for Developers》课程
  - 介绍：如何编写 Prompt:
    - 第一个方面：编写清晰、具体的指令
    - 第二个方面：给模型些时间思考
- [《ChatGPT Prompt Engineering for Developers》 学习 之 如何 优化 Prompt？](https://articles.zsxq.com/id_swisqrpd8vi2.html)
  - [吴恩达老师与OpenAI合作推出《ChatGPT Prompt Engineering for Developers》](https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/1/introduction)
  - 动机：吴恩达老师与OpenAI合作推出《ChatGPT Prompt Engineering for Developers》课程
  - 介绍：优化编写好 Prompt
- [《ChatGPT Prompt Engineering for Developers》 学习 之 如何使用 Prompt 处理 NLP特定任务？](https://articles.zsxq.com/id_uazuhqdkesq4.html)
  - [吴恩达老师与OpenAI合作推出《ChatGPT Prompt Engineering for Developers》](https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/1/introduction)
  - 动机：吴恩达老师与OpenAI合作推出《ChatGPT Prompt Engineering for Developers》课程
  - 介绍：如何构建ChatGPT Prompt以处理文本摘要、推断和转换(翻译、纠错、风格转换、格式转换等)这些常见的NLP任务

### 第九层 LLMs to interview

- [LLMs 降龙十八掌](https://articles.zsxq.com/id_un190pd3gi8a.html)
  - LLMs Fine-Tuning 经验贴
    - 如何解决大模型遗忘问题#ChatGPT# #LLM (大型语言模型)
    - Fine-Tuning max_length 选择 问题
    - Fine-Tuning Learning rate 选择 问题
    - 如何 向GPT/LLM模型添加额外知识？
    - 如何解决LLM大模型fine-tune过拟合问题？
    - 英文模型需要做词表扩充吗？
    - 全参finetune，数据配比?
    - 百川13b强化学习时遇到参数不更新的问题?
    - 使用lion优化器节省显存?
    - 使用lora训练的参数配置?
    - ...
  - LLMs 推理 经验贴
    - chatglm微调完针对相同的问题输出都是一样的，有办法让他随机一点吗？
    - 使用fastllm加速推理
  - LLMs 部署 经验贴
    - 如果只跑一个私有大模型，但是很多用户同时访问。这些针对不同用户的chat history怎么存取比较高效？
  - LLMs 多轮对话 经验贴
    - 如何 融入 之前对话信息
  - 数据构建篇
    - 训练数据如何构建？
  - LLMs 胡思乱想
    - 为什么 LLMs 不容易出现 灾难性遗忘问题？
    - LLMs to NLP 下游任务 经验贴

- [【LLMs】大模型 千面郎君](https://articles.zsxq.com/id_u51mg6wdo4h5.html)
  - 大模型常识面
    - 简单 介绍一下 大模型【LLMs】？
    - 大模型【LLMs】后面跟的 175B、60B、540B等 指什么？
    - 大模型【LLMs】具有什么优点？
    - 大模型【LLMs】具有什么缺点？
  - 大模型强化学习面
    - 简单介绍强化学习？
    - 简单介绍一下 RLHF？
  - 大模型【LLMs】微调篇
    - 大模型【LLMs】泛化问题？
    - 大模型【LLMs】微调问题？
    - 大模型【LLMs】微调有哪些优点？
    - 大模型【LLMs】指令微调问题？
  - 大模型【LLMs】思维链篇
    - 大模型【LLMs】思维链问题？
    - 大模型【LLMs】思维链本质是什么？
    - 大模型【LLMs】思维链优点是什么？
    - 大模型【LLMs】思维链类型和策略？
    - 大模型【LLMs】逐步Zero-shot 介绍？
    - 大模型【LLMs】Zero-shot-CoT提示策略定义？
    - 大模型【LLMs】Zero-shot-CoT提示策略应用方法？
    - 大模型【LLMs】Few-shot-CoT提示策略定义？
    - 大模型【LLMs】Few-shot-CoT提示策略核心思想是什么？
    - 大模型【LLMs】Few-shot-CoT提示策略应用方法是什么？
  - 大模型【LLMs】涌现现象篇
    - 大模型【LLMs】中有一种 涌现现象，你知道么？
    - 大模型【LLMs】涌现现象主要体现在哪些方面？
    - 大模型【LLMs】涌现现象主激活方式？
  - 大模型【LLMs】提示工程篇
    - 大模型【LLMs】提示工程 是什么？
    - 提示工程 如何添加进 大模型【LLMs】？
    - 微调（FineTuning） vs 提示工程？
    - 微调（FineTuning） vs 提示工程 在应用场景中关系雨联系？
    - 大模型【LLMs】Few-shot提示方法 是什么？

- [大模型的重复生成现象如何缓解？](https://articles.zsxq.com/id_u9pxnveolxwm.html)
  - 为何会出现重复生成现象？
  - 如何减少大模型重复生成？
- [ LoRA这种微调方法和全参数比起来有什么劣势吗？](https://articles.zsxq.com/id_6dbtb0w32qo3.html)
  - 什么是 LoRA？
  - LoRA 优点是什么？
  - LoRA 缺点是什么？
  - LoRA这种微调方法和全参数比起来有什么劣势吗？
  - LoRA这种微调方法和全参数 如何选择？
- [LLaMa-1 从原理到实践分析](https://articles.zsxq.com/id_sil90pxf3zvw.html)
  - LLaMa-1 理论介绍
  - LLaMa 模型架构介绍
  - LLaMa 优化器介绍
  - LLaMa-1 实践介绍
- [LLaMa-2 从原理到实践分析](https://articles.zsxq.com/id_gsm9rxffc2r8.html)
  - LLaMa-2 理论介绍
  - LLaMa-2 实践介绍
- [基于PyTorch来优化大模型训练的内存（显存）](https://articles.zsxq.com/id_vri8pu8oacqu.html)
  - 动机：峰值内存（显存）的消耗直接决定了 机器 是否 支撑大模型训练
  - LLMs Fine-Tuning 经验贴
    - LLMs Fine-Tuning 框架依赖问题？
    - LLMs Fine-Tuning 显存问题？
    - 如何解决大模型遗忘问题#ChatGPT# #LLM (大型语言模型)？
    - ...
  - 混合精度训练（Mixed-Precision Training）
    - 为什么需要 混合精度训练（Mixed-Precision Training）？
    - 什么是 混合精度训练（Mixed-Precision Training）？
    - 混合精度训练（Mixed-Precision Training）实现步骤？
    - ...
  - 低精度训练（Lower-Precision Training）
    - 为什么需要 低精度训练（Lower-Precision Training）？
    - 什么是 低精度训练（Lower-Precision Training）？
    - ...
  - 降低训练批处理大小（Reducing the Batchsize）
    - 为什么需要 降低训练批处理大小（Reducing the Batchsize）？
    - 什么是 降低训练批处理大小（Reducing the Batchsize）？
    - ...
  - 使用梯度累积创建微批次（Using Gradient Accumulation to Create Microbatches）
    - 为什么需要 使用梯度累积创建微批次（Using Gradient Accumulation to Create Microbatches）？
    - 什么是 使用梯度累积创建微批次（Using Gradient Accumulation to Create Microbatches）？
    - ...
- [如何缓解大模型幻觉？](https://articles.zsxq.com/id_bxani4mwdead.html)
  - 为什么 会 出现 大模型幻觉？
  - 如何 缓解 大模型幻觉？

### 第八层 LLMs to Inference acceleration

- [LLM（大语言模型）部署加速方法——PagedAttention](https://articles.zsxq.com/id_7eu43rgc4ehm.html)
  - 什么是 PagedAttention？
  - PagedAttention 如何存储 连续的key和value？
  - PagedAttention 技术细节？
  - PagedAttention 如何 实现安全共享？
  - PagedAttention 源码介绍？

- [LLM（大语言模型）部署加速方法——Faster Transformer](https://articles.zsxq.com/id_2ixvwo53necu.html)
  - 为什么需要 Faster Transformer？
  - 什么是 FasterTransformer？
  - FasterTransformer 核心？
  - FasterTransformer 优化？

- [纯Python超轻量高性能LLM推理框架 —— LightLLM](https://articles.zsxq.com/id_jwdwtitwtjwj.html)
  - 为什么 需要 LightLLM ?
    - 显存碎片化严重
    - 请求调度效率低
    - kernel定制化难度高
  - 介绍：基于纯Python语言的大模型推理部署框架LightLLM，方便研究员进行轻量级的本地部署和定制修改，用于快速扩展对不同模型的支持，吸纳层出不穷的优秀开源特性，探索最优服务架构。
  - LightLLM 性能表现
    - TGI由于显存碎片化严重，所以很难达到较高的吞吐量；
    - vLLM因引入了PageAttention，但是由于整体实现细节更利于小模型推理，所以在大模型上的并发性能并不是十分理想（使用的默认配置）；
    - 相比之下，LightLLM则可以在各种大小的模型下都保持稳健的性能，在大模型上（LLaMA-65B）相对TGI和vLLM实现了3倍左右的2提升。

- [大模型推理加速工具 —— vLLM](https://articles.zsxq.com/id_ra748ubp2t3l.html)
  - 介绍：vLLM是一个开源的LLM推理和服务引擎。它利用了全新的注意力算法「PagedAttention」，有效地管理注意力键和值。
  - vLLM 具有哪些特点 ?
    - 最先进的服务吞吐量；
    - PagedAttention 可以有效的管理注意力的键和值；
    - 动态批处理请求；
    - 优化好的 CUDA 内核；

### 第二层 LLMs to Parameter Efficient Fine-Tuning (PEFT)

#### 第一重 分布式训练神器

- [分布式训练神器 之 ZeRO 学习](trick/distributed/ZeRO.md)
  - 动机：虽然 DataParallel (DP) 因为简单易实现，所以目前应用相比于其他两种 广泛，但是 由于 DataParallel (DP) 需要 每张卡都存储一个模型，导致 显存大小 成为 制约模型规模 的 主要因素。
  - 核心思路：去除数据并行中的冗余参数，使每张卡只存储一部分模型状态，从而减少显存占用。

- [图解分布式训练（一） —— 流水线并行（Pipeline Parallelism）](https://articles.zsxq.com/id_5116ludvwvww.html)
  - 动机：
    - 回顾ChatGPT的发展历程，我们可以总结出大语言模型（LLM）取得惊艳效果的要点（重要性从高到低排序）：
      - 愿意烧钱，且接受“烧钱 != 好模型”的现实
      - 高质量的训练语料
      - **高效的分布式训练框架**和**充沛优质的硬件资源**
      - **算法的迭代创新**
  - 介绍：在大模型训练这个系列里，我们将一起探索学习几种经典的分布式并行范式，包括**流水线并行（Pipeline Parallelism）**，**数据并行（Data Parallelism）**和**张量并行（Tensor Parallesim）**。

- [图解分布式训练（二） —— nn.DataParallel](https://articles.zsxq.com/id_m3tdrv0wdlzr.html)
  - 动机：
    - 多GPU并行训练的原理就是将模型参数和数据分布到多个GPU上，同时利用多个GPU计算加速训练过程。具体实现需要考虑以下两个问题：
    - 数据如何划分？因为模型需要处理的数据通常很大，将所有数据放入单个GPU内存中可能会导致内存不足，因此我们需要将数据划分到多个GPU上。
  - 介绍：
    - 计算如何协同？因为每个GPU都需要计算模型参数的梯度并将其发送给其他GPU，因此需要使用同步机制来保证计算正确性。一般有两种同步方式：
    - 数据同步：在每个GPU上计算模型参数的梯度，然后将梯度发送到其他GPU上进行汇总，最终更新模型参数。
    - 模型同步：在每个GPU上计算模型参数的梯度，然后将模型参数广播到其他GPU上进行汇总，最终更新模型参数。

- [图解分布式训练（三） ——  nn.parallel.DistributedDataParallel](https://articles.zsxq.com/id_w11rmbn6i9zl.html)
  - 动机：
    - **数据如何划分？**因为模型需要处理的数据通常很大，将所有数据放入单个GPU内存中可能会导致内存不足，因此我们需要将数据划分到多个GPU上。一般有两种划分方式：
      - 数据并行：将数据分割成多个小批次，每个GPU处理其中的一个小批次，然后将梯度汇总后更新模型参数。
      - 模型并行：将模型分解成多个部分，每个GPU处理其中一个部分，并将处理结果传递给其他GPU以获得最终结果。
    - **计算如何协同？**因为每个GPU都需要计算模型参数的梯度并将其发送给其他GPU，因此需要使用同步机制来保证计算正确性。一般有两种同步方式：
      - 数据同步：在每个GPU上计算模型参数的梯度，然后将梯度发送到其他GPU上进行汇总，最终更新模型参数。
      - 模型同步：在每个GPU上计算模型参数的梯度，然后将模型参数广播到其他GPU上进行汇总，最终更新模型参数。
    - DP 只支持 单机多卡场景，在 多机多卡 场景 下，DP 的 通讯问题将被放大:
      - DDP首先要解决的就是通讯问题：将Server上的通讯压力均衡转到各个Worker上。实现这一点后，可以进一步去Server，留Worker。
  - 介绍：上节讲到 DP 只支持 单机多卡场景，主要原因是 DP 无法数据并行中通讯负载不均的问题， 而 DDP 能够解决 该问题 的 核心在于 **Ring-AllReduce**。它由百度最先提出，非常有效地解决了数据并行中通讯负载不均的问题，使得DDP得以实现。

- [图解分布式训练（四） ——  torch.multiprocessing 详细解析](trick/distributed/multiprocessing.md)
  - 介绍：torch.multiprocessing是本机multiprocessing模块的封装。封装了multiprocessing模块。它注册自定义的reducer，它使用共享内存为不同进程中的相同数据提供视图共享。一旦张量/存储被移动到shared_memory（参见sharememory()），就可以将其发送到其他进程而不进行其它任何操作。

- [图解分布式训练（五） ——  AMP混合精度训练 详细解析](trick/distributed/AMP.md)
  - 动机：PyTorch 1.6版本今天发布了，带来的最大更新就是自动混合精度。
  - 介绍：在某些上下文中torch.FloatTensor有优势，在某些上下文中torch.HalfTensor有优势呗。答案进一步可以转化为，相比于之前的默认的torch.FloatTensor，torch.HalfTensor有时具有优势，有时劣势不可忽视。

- [图解分布式训练（六） ——  Pytorch的 DeepSpeed 详细解析](trick/distributed/deepspeed.md)
  - 动机：
    - 最常见的深度学习框架应该是TensorFlow、Pytorch、Keras，但是这些框架在面向大规模模型的时候都不是很方便。
    - 比如Pytorch的分布式并行计算框架（Distributed Data Parallel，简称DDP），它也仅仅是能将数据并行，放到各个GPU的模型上进行训练。
    - 也就是说，DDP的应用场景在你的模型大小大于显卡显存大小时，它就很难继续使用了，除非你自己再将模型参数拆散分散到各个GPU上。
  - 介绍：在分布式计算环境中，需要理解几个非常基础的概念：节点编号、全局进程编号、局部进程编号、全局总进程数和主节点。其中，主节点负责协调所有其他节点和进程的工作，因此是整个系统的关键部分。

- [图解分布式训练（七） ——  accelerate 分布式训练 详细解析](trick/distributed/accelerate.md)
  - 介绍：PyTorch Accelerate 是一个 PyTorch 的加速工具包，旨在简化 PyTorch 训练和推断的开发过程，并提高性能。它是由 Hugging Face、NVIDIA、AWS 和 Microsoft 等公司联合开发的，是一个开源项目。

#### 第二重 LLMs Trick

- [LLMs Trick](trick/readme.md)

#### 第三重 LLMTune

- [【LLMs 入门实战 —— 十六 】 LLMTune 模型学习与实战](https://articles.zsxq.com/id_1hg51c292bu6.html)
  - 动机：大语言模型虽然能力很强，目前开源生态也很丰富，但是在特定领域微调大模型依然需要大规格的显卡。例如，清华大学发布的ChatGLM-6B，参数规模60亿，在没有量化的情况下微调需要14GB显存（parameter-efficient fine-tuning，PEFT)。在没有任何优化的前提下，每10亿参数的全精度（32bit）模型载入到显存中就需要4GB，而int8量化后也需要1GB显存。而目前开源最强的模型LLaMA，其最高参数维650亿规模，全精度模型载入就需要260GB，显然已经超出了大部分人的硬件水平。更不要说对模型进行微调（微调需要训练更新参数，推理只需要前向计算即可，因此，微调需要更多的显存才能支持）。
  - 介绍：Cornell Tech开源的LLMTune就是为了降低大模型微调难度所提出的一种解决方案。对于650亿参数的LLaMA模型微调仅需要40GB显存即可。
  - github 地址: https://github.com/kuleshov-group/llmtune

#### 第四重 QLoRA

- [【LLMs 入门实战 —— 二十 】 QLoRA 模型学习与实战]()
  - [https://huggingface.co/BlinkDL](https://huggingface.co/BlinkDL)
  - [artidoro/qlora](https://github.com/artidoro/qlora)
  - 模型：[timdettmers (Tim Dettmers)](https://huggingface.co/timdettmers)
  - 量化代码：[TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
  - BLOG : [Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
  - Demo环境：[Guanaco Playground Tgi - a Hugging Face Space by uwnlp](https://huggingface.co/spaces/uwnlp/guanaco-playground-tgi)
  - 介绍：5月24日华盛顿大学的研究者发布了QLoRA技术及用其生成的Guanaco大模型。
    - 特点：
      - 在Vicuna基准测试中表现优于所有先前公开发布的模型，达到ChatGPT性能水平的99.3%，仅需要单个GPU上的24小时微调时间；
      - QLORA引入了一些创新来节省内存而不牺牲性能：
        - （a）4位NormalFloat（NF4），这是一种对于正态分布权重来说在信息论上是最优的数据类型；
        - （b）双量化，通过量化量化常数来减少平均内存占用；
        - （c）分页优化器，用于管理内存峰值。

- [【LLMs 入门实战 —— 二十七 】【QLoRA实战】使用单卡高效微调bloom-7b1]()
  - [https://huggingface.co/BlinkDL](https://huggingface.co/BlinkDL)
  - [artidoro/qlora](https://github.com/artidoro/qlora)
  - 模型：[timdettmers (Tim Dettmers)](https://huggingface.co/timdettmers)
  - 量化代码：[TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
  - BLOG : [Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

- [【LLMs 入门实战 】Anima 学习与实战](Anima/readme.md)
  - Github 代码：https://github.com/lyogavin/Anima
  - 模型：https://huggingface.co/lyogavin/Anima33B
  - 动机：之前大部分开源可finetune的模型大都是比较小的模型7B或者13B，虽然可以在一些简单的chatbot评测集上，通过finetune训练有不错的表现。但是由于这些模型规模还是有限，LLM核心的reasoning的能力还是相对比较弱。这就是为什么很多这种小规模的模型在实际应用的场景表现像是个玩具。chatbot评测集比较简单，真正比较考验模型能力的复杂逻辑推理及数学问题上小模型和大模型差距还是很明显的。
  - 介绍：QLoRA的优化方法，第一次让33B规模的模型可以比较民主化的，比较低成本的finetune训练，让33B模型的普及使用成为了可能。我们认为33B模型既可以发挥大规模模型的比较强的推理能力，又可以针对私有业务领域数据进行灵活的finetune训练提升对于LLM的控制力。

### 第三层 LLMs to Artifact

#### 第一重 langchain

- [【LLMs 入门实战 —— 十二 】基于 本地知识库 的高效 🤖langchain-ChatGLM ](https://articles.zsxq.com/id_54vjwns5t6in.html)
  - 介绍：langchain-ChatGLM是一个基于本地知识的问答机器人，使用者可以自由配置本地知识，用户问题的答案也是基于本地知识生成的。
- [【LLMs 入门实战 —— 三十一 】Vicuna-LangChain 模型学习与实战](https://articles.zsxq.com/id_74t8mmww5a1n.html)
  - 介绍：一个简单的类LangChain实现，基于Sentence Embedding+本地知识库，以Vicuna作为生成模型。支持中英双语，支持pdf、html和docx格式的文档作为知识 库。
  - Vicuna-LangChain 思路
    1. 提取知识库文件夹中的文档文本，分割成chunk_length大小的文本块
    2. 通过shibing624/text2vec-base-chinese模型计算各文本块的嵌入
    3. 计算问题文本嵌入和各文本块的嵌入的余弦相似度
    4. 返回余弦相似度最高的k个文本作为给定信息生成prompt
    5. 将prompt历史替换为最初问的问题
    6. 将prompt交给vicuna模型生成答案
- [【LLMs 入门实战】基于 本地知识库 的高效 🤖langchain-ChatGLM2 ](https://articles.zsxq.com/id_zx9d1pix6wls.html)
  - 介绍：langchain-ChatGLM2是一个基于本地知识的问答机器人，使用者可以自由配置本地知识，用户问题的答案也是基于本地知识生成的。

#### 第二重 wenda

- [【LLMs 入门实战】基于 本地知识库 的高效 🤖wenda+ChatGLM2-6B](https://articles.zsxq.com/id_mx6n1d28iccg.html)
  - 介绍：本项目设计目标为实现针对特定环境的高效内容生成，同时考虑个人和中小企业的计算资源局限性，以及知识安全和私密性问题。为达目标，平台化集成了以下能力：
    - 知识库：支持对接本地离线向量库、本地搜索引擎、在线搜索引擎等。
    - 多种大语言模型：目前支持离线部署模型有chatGLM-6B\chatGLM2-6B、chatRWKV、llama系列(不推荐中文用户)、moss(不推荐)、baichuan(需配合lora使用，否则效果差)、Aquila-7B，在线API访问openai api和chatGLM-130b api。
    - Auto脚本：通过开发插件形式的JavaScript脚本，为平台附件功能，实现包括但不限于自定义对话流程、访问外部API、在线切换LoRA模型。
    - 其他实用化所需能力：对话历史管理、内网部署、多用户同时使用等。

#### 第三重 AutoGPT

- [AutoGPT 使用和部署](https://articles.zsxq.com/id_pli0z9916126.html)
  - 介绍：Auto-GPT是一个基于ChatGPT的工具，他能帮你自动完成各种任务，比如写代码、写报告、做调研等等。使用它时，你只需要告诉他要扮演的角色和要实现的目标，然后他就会利用ChatGPT和谷歌搜索等工具，不断“思考”如何接近目标并执行，你甚至可以看到他的思考过程。

#### 第四重 Knowledge Extraction

- [【LLMs 入门实战 】 DeepKE-LLM 模型学习与实战]()
  - DeepKE-LLM链接：https://github.com/zjunlp/DeepKE/tree/main/example/llm
  - OpenKG地址：http://openkg.cn/tool/deepke
  - Gitee地址：https://gitee.com/openkg/deepke/tree/main/example/llm
  - 介绍：DeepKE是一个开源可扩展的知识图谱抽取工具，可实现命名实体识别、关系抽取和属性抽取等抽取任务，并支持低资源少样本、文档篇章和多模态等复杂场景。在大模型时代，我们对DeepKE进行了全面升级并发布大模型版DeepKE-LLM（智析抽取大模型）。该版本基于大模型对数据进行智能解析以实现知识抽取，支持多种大模型、遵循指令和自定义知识类型、格式。

### 第四层 LLMs to Text-to-Image

#### 第一重 Stable Diffusion

- [【LLMs 入门实战 —— 二十二 】Stable Diffusion 模型学习与实战](https://github.com/km1994/LLMsNineStoryDemonTower/tree/main/text2img/stable_diffusion)
  - Github 地址：https://github.com/gediz/lstein-stable-diffusion
  - 预训练模型：https://huggingface.co/CompVis/stable-diffusion
  - 介绍：Stable Diffusion是一种潜在扩散模型（Latent Diffusion Model），能够从文本描述中生成详细的图像。它还可以用于图像修复、图像绘制、文本到图像和图像到图像等任务。简单地说，我们只要给出想要的图片的文字描述在提Stable Diffusion就能生成符合你要求的逼真的图像！
- [【LLMs 入门实战 —— 二十三 】Stable Diffusion Webui 模型学习与实战](https://github.com/km1994/LLMsNineStoryDemonTower/tree/main/text2img/stable_diffusion_webui)
  - Github 地址：https://github.com/AUTOMATIC1111/stable-diffusion-webui
  - 预训练模型：https://huggingface.co/CompVis/stable-diffusion
  - 介绍：Stable Diffusion是一款功能异常强大的AI图片生成器。它不仅支持生成图片，使用各种各样的模型来达到你想要的效果，还能训练你自己的专属模型。WebUI使得Stable Diffusion有了一个更直观的用户界面，更适合新手用户。
- [【LLMs 入门实战 —— 二十四 】Novelai 模型学习与实战](https://github.com/km1994/LLMsNineStoryDemonTower/tree/main/text2img/novelai)
- [【LLMs 入门实战 —— 二十五 】lora 模型学习与实战](https://github.com/km1994/LLMsNineStoryDemonTower/tree/main/text2img/lora)
  - Github 地址：https://github.com/microsoft/LoRA
  - 预训练模型：https://huggingface.co/johnsmith007/LoRAs/tree/main
  - 介绍：LoRA的全称是LoRA: Low-Rank Adaptation of Large Language Models，可以理解为stable diffusion（SD)模型的一种插件，和hyper-network，controlNet一样，都是在不修改SD模型的前提下，利用少量数据训练出一种画风/IP/人物，实现定制化需求，所需的训练资源比训练SD模要小很多，非常适合社区使用者和个人开发者。

### 第五层 LLMs to Visual Question Answering (VQA)

#### 第一重 BLIP

- [【LLMs 入门实战 —— 二十二】 BLIP 模型学习与实战](https://articles.zsxq.com/id_mpckrd9ccfdn.html)
  - 论文名称：BLIP: Bootstrapping Language-Image Pre-training for Uniﬁed Vision-Language Understanding and Generation
  - 论文地址：https://arxiv.org/abs/2201.12086
  - 代码地址：https://github.com/salesforce/BLIP
  - 局限性:
    - **模型角度**: 
      - 现有方法：大多数方法要么采用基于编码器的模型，要么采用编码器-解码器模型。
      - 存在问题：**基于编码器的模型不太容易直接转换到文本生成任务（例如图像字幕）**，而**编码器-解码器模型尚未成功用于图像文本检索任务**；
    - **数据角度**: 大多数SOTA的方法（如CLIP、ALBEF等）都在从web上收集到的图文对上进行预训练。尽管通过扩展数据集获得了性能提升，但 BLIP 的研究表明，对于视觉语言学习来说，有噪声的网络文本是次优的。
  - BLIP总体思路：作为新的 VLP 框架，**BLIP 用于统一视觉语言理解和生成的 Bootstrapping Language-Image 预训练，可以灵活地迁移到视觉语言理解和生成任务。 BLIP 通过引导字幕有效地利用了嘈杂的网络数据，字幕生成器生成合成字幕，过滤器去除嘈杂的字幕**
  - 贡献:
    - （1） **编码器-解码器 (MED) 的多模式混合**：一种用于有效多任务预训练和灵活迁移学习的新模型架构。MED可以作为单模态编码器、基于图像的文本编码器或基于图像的文本解码器工作。**该模型与三个视觉语言目标联合预训练：图像文本对比学习、图像文本匹配和图像条件语言建模**。
    - （2） **字幕和过滤（Captioning and Filtering，CapFilt）**：一种新的数据集增强方法，用于**从噪声图像-文本对中学习**。作者将预先训练的MED分为两个模块: **一个字幕器，用于生成给定web图像的合成字幕**，以及**一个过滤器，用于从原始web文本和合成文本中删除嘈杂的字幕**。

#### 第二重 BLIP2

- [【LLMs 入门实战 —— 二十六】 BLIP2 模型学习与实战](https://articles.zsxq.com/id_lcwp2s597vrr.html)
  - 论文名称：BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
  - 单位：Salesforce 研究院
  - 论文地址：https://arxiv.org/abs/2301.12597
  - 代码地址：https://github.com/salesforce/LAVIS/tree/main/projects/blip2
  - HF上的Demo：https://huggingface.co/spaces/Salesforce/BLIP2
  - 动机
    - 由于大规模模型的端到端训练，视觉和语言预训练的成本变得越来越高
    - 为了降低计算成本并抵消灾难性遗忘的问题，希望在 Vision-language pre-training (VLP) 中固定视觉模型参数与语言模型参数。然而，由于语言模型在其单模态预训练期间没有看到图像，因此冻结它们使得视觉语言对齐尤其具有挑战性
   - 介绍：
     - BLIP-2， 一种通用而有效的预训练策略，它从现成的冻结预训练图像编码器和冻结的大型语言模型中引导视觉语言预训练。
     - 通过一个轻量级的 Querying Transformer （Q-Former是一个轻量级的 transformer，它使用一组可学习的查询向量来从冻结图像编码器中提取视觉特征，为LLM提供最有用的视觉特征，以输出所需的文本） 弥补了模态 gap，该 Transformer 分两个阶段进行预训练:
      - 第一阶段：从冻结图像编码器引导视觉语言表示学习，强制 Q-Former 学习与文本最相关的视觉表示；
      - 第二阶段：将视觉从冻结的语言模型引导到语言生成学习，将Q-Former的输出连接到冻结的LLM，并对Q-Former进行训练，使其输出视觉表示能够被LLM解释。

#### 第三重 MiniGPT-4 

- [【LLMs 入门实战 —— 八 】MiniGPT-4 模型学习与实战](https://articles.zsxq.com/id_ff0w6czthq25.html)
  - Github 链接： https://github.com/Vision-CAIR/MiniGPT-4
  - 介绍： MiniGPT-4，是来自阿卜杜拉国王科技大学的几位博士做的，它能提供类似 GPT-4 的图像理解与对话能力

#### 第四重 VisualGLM-6B

- [【LLMs 入门实战 —— 十七 】 VisualGLM-6B 模型学习与实战](https://articles.zsxq.com/id_4pzgwnwl2zjc.html) 
  - Github 链接： https://github.com/THUDM/VisualGLM-6B
  - Huggingface 链接：https://huggingface.co/THUDM/visualglm-6b
  - 动机：OpenAI 的GPT-4样例中展现出令人印象深刻的多模态理解能力，但是能理解图像的中文开源对话模型仍是空白。
  - 介绍：VisualGLM-6B 是一个开源的，支持图像、中文和英文的多模态对话语言模型，语言模型基于 ChatGLM-6B，具有 62 亿参数；图像部分通过训练 BLIP2-Qformer 构建起视觉模型与语言模型的桥梁，整体模型共 78 亿参数。VisualGLM-6B 依靠来自于 CogView 数据集的30M高质量中文图文对，与 300M 经过筛选的英文图文对进行预训练，中英文权重相同。该训练方式较好地将视觉信息对齐到 ChatGLM 的语义空间；之后的微调阶段，模型在长视觉问答数据上训练，以生成符合人类偏好的答案。
  - github 地址:https://github.com/THUDM/VisualGLM-6B

#### 第五重 Ziya-Visual

- [【LLMs 入门实战 】 Ziya-Visual 模型学习与实战](https://articles.zsxq.com/id_f3wugvkt4w00.html) 
  - Ziya-Visual模型开源地址：https://huggingface.co/IDEA-CCNL/Ziya-BLIP2-14B-Visual-v1L/Ziya-BLIP2-14B-Visual-v1
  - Demo体验地址：https://huggingface.co/spaces/IDEA-CCNL/Ziya-BLIP2-14B-Visual-v1-DemoDEA-CCNL/Ziya-BLIP2-14B-Visual-v1-Demo
  - Ziya开源模型：https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1L/Ziya-LLaMA-13B-v1
  - 封神榜项目主页：https://github.com/IDEA-CCNL/Fengshenbang-LM
  - 介绍：自从3月份OpenAI发布具有识图能力的多模态大模型GPT-4，大模型的能力便不再局限于文本输入-文本输出的形式，人们可以上传视觉图片来与大模型进行聊天和交互。遗憾的是，时至今日绝大部分用户也都还没有拿到GPT-4输入图片的权限，无法体验到结合视觉和语言两大模态的大模型的卓越能力，而且GPT-4也没有叙述或者开源GPT模型多模态预训练的方案。与之相对的是，学术界和开源界则充分探索了视觉预训练模型（比如ViT, Vision Transformer）与大语言模型(LLM，Large Language Model)结合，从而让目前的LLM获得输入图片、认识图片的能力。其中的代表工作包括国外团队开源的Mini-GPT4[1]，LLaVA[2]等，国内团队开源的VisuaGLM[3]，mPLUG-Owl[4]等工作。大部分的开源方案参考了BLIP2的训练方案[5]，选择冻结LLM部分的参数训练或者采用Lora等parameter-efficient的微调训练方式。IDEA研究院封神榜团队在5月17日发布“姜子牙通用大模型v1”之后，继续发布Ziya-BLIP2-14B-Visual-v1多模态大模型（以下简称Ziya-Visual模型）。和Ziya大模型一样，Ziya-Visual模型具备中英双语能力，特别是中文能力较为突出。和所有基于BLIP2的方案类似，我们简单高效的扩展了LLM的识图能力。该模型对比VisualGLM、mPLUG-Owl模型，在视觉问答（VQA）评价和GPT-4打分评价[2]中，展现了一些优势。

### 第六层 LLMs to Automatic Speech Recognition (ASR)

#### 第一重 Massively Multilingual Speech (MMS，大规模多语种语音)

- [【LLMs 入门实战 —— 二十 】 Massively Multilingual Speech (MMS，大规模多语种语音) 模型学习与实战](https://github.com/km1994/LLMsNineStoryDemonTower/tree/main/speech_MMS_21)
  - 论文：[Scaling Speech Technology to 1,000+ Languages](https://research.facebook.com/publications/scaling-speech-technology-to-1000-languages/)
  - 代码：[fairseq/tree/main/examples/mms](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)
  - 公告：https://ai.facebook.com/blog/multilingual-model-speech-recognition/
  - 介绍：Meta 在 GitHub 上再次开源了一款全新的 AI 语言模型——Massively Multilingual Speech (MMS，大规模多语种语音)，它与 ChatGPT 有着很大的不同，这款新的语言模型可以识别 4000 多种口头语言并生成 1100 多种语音（文本到语音）。

#### 第二重 whisper 

- [【LLMs 入门实战】 whisper 模型学习与实战](whisper/readme.md)
  - Introducing Whisper: https://openai.com/blog/whisper/
  - Robust Speech Recognition via Large-Scale Weak Supervision: https://cdn.openai.com/papers/whisper.pdf
  - openai/whisper: https://github.com/openai/whisper
  - Huggin一 Face地址：Whisper - a Hugging Face Space by openai:https://huggingface.co/spaces/openai/whisper
  - Colab地址：Google Colaboratory: https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb
  - 介绍：Whisper这个模型是OpenAI在今年九月释放出来的吧（2022/09/21）Introducing Whisper ，不过这个效果真的是太好了，它的泛化性能跟别的模型一比，真是天上地下啊。
- [【LLMs 入门实战】 Faster-Whisper 模型学习与实战](whisper/FasterWhisper.md) 
  - Making OpenAI Whisper faster：https://nikolas.blog/making-openai-whisper-faster/
  - Faster Whisper transcription with CTranslate2 ：https://github.com/guillaumekln/faster-whisper
  - sanchit-gandhi/whisper-jax#creating-an-endpoint：https://github.com/sanchit-gandhi/whisper-jax#creating-an-endpoint
  - 介绍：Faster-Whisper是一种基于深度学习的语音识别模型，它采用了一种新颖的架构，可以在保持准确性的同时提高速度。Faster-Whisper的设计灵感来自于目标检测领域的Faster R-CNN模型，它将语音信号转换为频谱图，并使用卷积神经网络（CNN）和循环神经网络（RNN）进行特征提取和序列建模。Faster-Whisper的主要优点是速度快、准确性高、可扩展性强，适用于大规模语音识别任务。
  - Faster-Whisper的架构包括三个主要组件：
    - 前端：将语音信号转换为频谱图，使用CNN进行特征提取。
    - 后端：使用RNN进行序列建模，输出每个时间步的概率分布。
    - 解码器：将概率分布转换为最终的文本输出。

### 第七层 LLMs to Text To Speech (TTS)

#### 第一重 Massively Multilingual Speech (MMS，大规模多语种语音)

- [【LLMs 入门实战 —— 二十 】 Massively Multilingual Speech (MMS，大规模多语种语音) 模型学习与实战](https://github.com/km1994/LLMsNineStoryDemonTower/tree/main/speech_MMS_21)
  - 论文：[Scaling Speech Technology to 1,000+ Languages](https://research.facebook.com/publications/scaling-speech-technology-to-1000-languages/)
  - 代码：[fairseq/tree/main/examples/mms](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)
  - 公告：https://ai.facebook.com/blog/multilingual-model-speech-recognition/
  - 介绍：Meta 在 GitHub 上再次开源了一款全新的 AI 语言模型——Massively Multilingual Speech (MMS，大规模多语种语音)，它与 ChatGPT 有着很大的不同，这款新的语言模型可以识别 4000 多种口头语言并生成 1100 多种语音（文本到语音）。

#### 第二重 Retrieval-based-Voice-Conversion

- [【LLMs 入门实战】 Retrieval-based-Voice-Conversion-WebUI 模型学习与实战]()
  - 名称：Retrieval-based-Voice-Conversion-WebUI
  - 论文地址：https://arxiv.org/abs/2305.18975
  - Github 代码：https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
  - 使用了RVC的实时语音转换: [w-okada/voice-changer](https://github.com/w-okada/voice-changer)
  - 使用了RVC变声器训练的人声转木吉他模型在线demo ：https://huggingface.co/spaces/lj1995/vocal2guitar
  - RVC人声转吉他效果展示视频 ：https://www.bilibili.com/video/BV19W4y1D7tT/
  - 介绍：一个基于VITS的简单易用的语音转换（变声器）框架
  - 特点
    - 使用top1检索替换输入源特征为训练集特征来杜绝音色泄漏
    - 即便在相对较差的显卡上也能快速训练
    - 使用少量数据进行训练也能得到较好结果(推荐至少收集10分钟低底噪语音数据)
    - 可以通过模型融合来改变音色(借助ckpt处理选项卡中的ckpt-merge)
    - 简单易用的网页界面
    - 可调用UVR5模型来快速分离人声和伴奏

- [【LLMs 入门实战】 kNN-VC 模型学习与实战]()
  - 论文名称：Voice Conversion With Just Nearest Neighbors (kNN-VC)
  - 论文地址：https://arxiv.org/abs/2305.18975
  - Github 代码：https://github.com/bshall/knn-vc

### LLaMA 衍生物系列

#### 第五重 GPT4ALL

- [【LLMs 入门实战 —— 八 】GPT4ALL 模型学习与实战](https://articles.zsxq.com/id_ff0w6czthq25.html)
  - 介绍：一个 可以在自己笔记本上面跑起来的  Nomic AI 的助手式聊天机器人，成为贫民家孩子的 福音！

#### 第十一重 OpenBuddy

- [【LLMs 入门实战 —— 二十八 】 OpenBuddy 模型学习与实战](https://articles.zsxq.com/id_g2tmn66fearf.html)
  - 论文名称：OpenBuddy - Open Multilingual Chatbot based on Falcon
  - github 地址：https://github.com/OpenBuddy/OpenBuddy
  - 动机：虽然目前 很多人 LLMs 层出不穷，但是他们并不能 在 多语言支持无缝衔接（eg: LLaMA 模型由于是用 英语训练，所以在 中文等其他语种上效果并不好）
  - 介绍：基于 Tii 的 Falcon 模型和 Facebook 的 LLaMA 模型构建，OpenBuddy 经过微调，包括扩展词汇表、增加常见字符和增强 token 嵌入。通过利用这些改进和多轮对话数据集，OpenBuddy 提供了一个强大的模型，能够回答各种语言的问题并执行翻译任务。

#### 第十二重 Baize

- [【LLMs 入门实战 —— 三十 】Baize 学习与实战](https://articles.zsxq.com/id_hftswats1oe8.html)
  - 论文名称：Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data
  - 论文地址：https://arxiv.org/abs/2304.01196
  - Github 代码：https://github.com/project-baize/baize-chatbot/blob/main/README.md
  - 模型：
    - baize-v2-7b 模型：https://huggingface.co/project-baize/baize-v2-7b
    - baize-v2-13b 模型：https://huggingface.co/project-baize/baize-v2-13b
  - baize 体验网站：https://huggingface.co/spaces/project-baize/baize-lora-7B
  - 动机：高质量的标注数据问题
  - 介绍：Baize 作者 提出了一个自动收集 ChatGPT 对话的流水线，通过从特定数据集中采样「种子」的方式，让 ChatGPT 自我对话，批量生成高质量多轮对话数据集。其中如果使用领域特定数据集，比如医学问答数据集，就可以生成高质量垂直领域语料。

#### 第十三重 OpenChineseLLaMA

- [【LLMs 入门实战 】OpenChineseLLaMA 学习与实战](https://articles.zsxq.com/id_kbaocwse89o9.html)
  - 论文名称：
  - 论文地址：
  - Github 代码：https://github.com/OpenLMLab/OpenChineseLLaMA
  - 模型：https://huggingface.co/openlmlab/open-chinese-llama-7b-patch
  - 介绍：基于 LLaMA-7B 经过中文数据集增量预训练产生的中文大语言模型基座，对比原版 LLaMA，该模型在中文理解能力和生成能力方面均获得较大提升，在众多下游任务中均取得了突出的成绩。

#### 第十四重 Panda

- [【LLMs 入门实战 】Panda 学习与实战](https://articles.zsxq.com/id_6dnefz5bmekd.html)
  - 论文名称：Panda LLM: Training Data and Evaluation for Open-Sourced Chinese Instruction-Following Large Language Models
  - 论文地址：https://arxiv.org/pdf/2305.03025v1.pdf
  - Github 代码：https://github.com/dandelionsllm/pandallm
  - 模型：
  - 介绍：开源了基于LLaMA-7B, -13B, -33B, -65B 进行中文领域上的持续预训练的语言模型, 使用了接近 15M 条数据进行二次预训练。

#### 第十五重 Ziya-LLaMA-13B

- [【LLMs 入门实战 】 Ziya-LLaMA-13B 学习与实战](https://articles.zsxq.com/id_110iedww8t0k.html)
  - 论文名称：
  - 论文地址：
  - Github 代码：https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1
  - 模型：
  - 介绍：该项目开源了姜子牙通用大模型V1，是基于LLaMa的130亿参数的大规模预训练模型，具备翻译，编程，文本分类，信息抽取，摘要，文案生成，常识问答和数学计算等能力。该模型已完成大规模预训练、多任务有监督微调和人类反馈学习三阶段的训练过程。

#### 第十六重 BiLLa

- [【LLMs 入门实战 】 BiLLa 学习与实战](https://articles.zsxq.com/id_wtixv8u69v7w.html)
  - 论文名称：BiLLa: A Bilingual LLaMA with Enhanced Reasoning Ability
  - 论文地址：
  - Github 代码：https://github.com/Neutralzz/BiLLa
  - 模型：
  - 介绍：该项目开源了推理能力增强的中英双语LLaMA模型。模型的主要特性有：较大提升LLaMA的中文理解能力，并尽可能减少对原始LLaMA英文能力的损伤；训练过程增加较多的任务型数据，利用ChatGPT生成解析，强化模型理解任务求解逻辑；全量参数更新，追求更好的生成效果。

#### 第十七重 Luotuo-Chinese-LLM

- [【LLMs 入门实战 】 Luotuo-Chinese-LLM 学习与实战]()
  - 论文名称：
  - 论文地址：
  - Github 代码：https://github.com/LC1332/Luotuo-Chinese-LLM
  - 模型：
  - 介绍：囊括了一系列中文大语言模型开源项目，包含了一系列基于已有开源模型（ChatGLM, MOSS, LLaMA）进行二次微调的语言模型，指令微调数据集等。

#### 第十八重 Linly

- [【LLMs 入门实战 】 Linly 学习与实战]()
  - 论文名称：
  - 论文地址：
  - Github 代码：https://github.com/CVI-SZU/Linly
  - 模型：
  - 介绍：提供中文对话模型 Linly-ChatFlow 、中文基础模型 Linly-Chinese-LLaMA 及其训练数据。中文基础模型以 LLaMA 为底座，利用中文和中英平行增量预训练。项目汇总了目前公开的多语言指令数据，对中文模型进行了大规模指令跟随训练，实现了 Linly-ChatFlow 对话模型。

#### 第十九重 ChatYuan

- [【LLMs 入门实战 】 ChatYuan 学习与实战]()
  - 论文名称：
  - 论文地址：
  - Github 代码：https://github.com/clue-ai/ChatYuan
  - 介绍：元语智能发布的一系列支持中英双语的功能型对话语言大模型，在微调数据、人类反馈强化学习、思维链等方面进行了优化。

#### 第二十重 CPM-Bee

- [【LLMs 入门实战 】 CPM-Bee 学习与实战]()
  - 论文名称：
  - 论文地址：
  - Github 代码：https://github.com/OpenBMB/CPM-Bee
  - 模型：
  - 介绍：一个完全开源、允许商用的百亿参数中英文基座模型。它采用Transformer自回归架构（auto-regressive），在超万亿（trillion）高质量语料上进行预训练，拥有强大的基础能力。开发者和研究者可以在CPM-Bee基座模型的基础上在各类场景进行适配来以创建特定领域的应用模型。

#### 第二十一重 TigerBot

- [【LLMs 入门实战 】 TigerBot 学习与实战]()
  - 论文名称：
  - 论文地址：
  - Github 代码：https://github.com/TigerResearch/TigerBot
  - 模型：
  - 介绍：一个多语言多任务的大规模语言模型(LLM)，开源了包括模型：TigerBot-7B, TigerBot-7B-base，TigerBot-180B，基本训练和推理代码，100G预训练数据，涵盖金融、法律、百科的领域数据以及API等。

#### 第二十二重 书生·浦语

- [【LLMs 入门实战 】 书生·浦语 学习与实战]()
  - 论文名称：
  - 论文地址：
  - Github 代码：https://github.com/InternLM/InternLM-techreport
  - 模型：
  - 介绍：商汤科技、上海AI实验室联合香港中文大学、复旦大学和上海交通大学发布千亿级参数大语言模型“书生·浦语”（InternLM）。据悉，“书生·浦语”具有1040亿参数，基于“包含1.6万亿token的多语种高质量数据集”训练而成。

#### 第二十三重 Aquila

- [【LLMs 入门实战 】 Aquila 学习与实战]()
  - 论文名称：
  - 论文地址：
  - Github 代码： https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila
  - 模型：
  - 介绍：由智源研究院发布，Aquila语言大模型在技术上继承了GPT-3、LLaMA等的架构设计优点，替换了一批更高效的底层算子实现、重新设计实现了中英双语的tokenizer，升级了BMTrain并行训练方法，是在中英文高质量语料基础上从０开始训练的，通过数据质量的控制、多种训练的优化方法，实现在更小的数据集、更短的训练时间，获得比其它开源模型更优的性能。也是首个支持中英双语知识、支持商用许可协议、符合国内数据合规需要的大规模开源语言模型。

#### 第一重 金融领域

- [【LLMs 入门实战 —— 十九】 聚宝盆(Cornucopia) 模型学习与实战](https://github.com/km1994/LLMsNineStoryDemonTower/tree/main/Cornucopia_19)
  - 聚宝盆(Cornucopia) 开源了经过中文金融知识指令精调/指令微调(Instruct-tuning) 的LLaMA-7B模型。通过中文金融公开数据+爬取的金融数据构建指令数据集，并在此基础上对LLaMA进行了指令微调，提高了 LLaMA 在金融领域的问答效果。
  - github: [jerry1993-tech/Cornucopia-LLaMA-Fin-Chinese](https://github.com/jerry1993-tech/Cornucopia-LLaMA-Fin-Chinese/tree/main)
- [【LLMs 入门实战 】 BBT-FinCUGE-Applications 学习与实战]()
  - 论文名称：
  - 论文地址：https://github.com/ssymmetry/BBT-FinCUGE-Applications
  - 介绍：开源了中文金融领域开源语料库BBT-FinCorpus，中文金融领域知识增强型预训练语言模型BBT-FinT5及中文金融领域自然语言处理评测基准CFLEB。
- [【LLMs 入门实战 】 XuanYuan（轩辕） 学习与实战]()：首个千亿级中文金融对话模型
  - 论文名称：
  - 论文地址：https://huggingface.co/xyz-nlp/XuanYuan2.0
  - 介绍：轩辕是国内首个开源的千亿级中文对话大模型，同时也是首个针对中文金融领域优化的千亿级开源对话大模型。轩辕在BLOOM-176B的基础上针对中文通用领域和金融领域进行了针对性的预训练与微调，它不仅可以应对通用领域的问题，也可以解答与金融相关的各类问题，为用户提供准确、全面的金融信息和建议。

#### 第二重 医疗领域

- [【LLMs 入门实战 —— 二十九 】HuatuoGPT (华佗GPT) 学习与实战](HuatuoGPT_29/readme.md)
  - HuatuoGPT (华佗GPT), Towards Taming Language Models To Be a Doctor.
  - 论文地址：https://arxiv.org/pdf/2305.15075.pdf
  - Github 代码：https://github.com/FreedomIntelligence/HuatuoGPT
  - 模型：https://huggingface.co/FreedomIntelligence/HuatuoGPT-7b-v1
  - HuatuoGPT 体验网站：https://www.huatuogpt.cn/
  - HuatuoGPT (华佗GPT) 监督微调（SFT）：[HuatuoGPT-sft-data-v1](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1)
  - 动机：
    - 益增长的在线和医院快速医疗咨询需求 与 医生的时间和精力 矛盾问题
    - 目前并没有 开源而且高质量的 可用于训练 medical LLMs，所以 为 训练 medical LLMs 而构建 high-quality instruction training data 至关重要；
    - medical LLMs 诊断能力需要进行 彻底评估和测试，避免 medical LLMs 误诊问题；
  - 介绍：
    - HuatuoGPT (华佗GPT) 知识库是一个在庞大的中国医学语料库上训练的大型语言模型。HuatuoGPT (华佗GPT) 的目标是为医疗咨询场景构建一个更专业的“ChatGPT”。

- [【LLMs 入门实战 】DoctorGLM 学习与实战]()
  - 论文名称：
  - 论文地址：https://github.com/xionghonglin/DoctorGLM
  - 介绍：基于 ChatGLM-6B的中文问诊模型，通过中文医疗对话数据集进行微调，实现了包括lora、p-tuningv2等微调及部署

- [【LLMs 入门实战 】 BenTsao 学习与实战]()
  - 论文名称：
  - 论文地址：https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese
  - 介绍：开源了经过中文医学指令精调/指令微调(Instruct-tuning) 的LLaMA-7B模型。通过医学知识图谱和GPT3.5 API构建了中文医学指令数据集，并在此基础上对LLaMA进行了指令微调，提高了LLaMA在医疗领域的问答效果。

- [【LLMs 入门实战 】 BianQue 学习与实战]()
  - 论文名称：
  - 论文地址：https://github.com/scutcyr/BianQue
  - 介绍：一个经过指令与多轮问询对话联合微调的医疗对话大模型，基于ClueAI/ChatYuan-large-v2作为底座，使用中文医疗问答指令与多轮问询对话混合数据集进行微调。

- [【LLMs 入门实战 】 Med-ChatGLM 学习与实战]()
  - 论文名称：
  - 论文地址：https://github.com/SCIR-HI/Med-ChatGLM
  - 介绍：基于中文医学知识的ChatGLM模型微调，微调数据与BenTsao相同。

- [【LLMs 入门实战 】 QiZhenGPT 学习与实战]()
  - 论文名称：
  - 论文地址：https://github.com/CMKRG/QiZhenGPT
  - 介绍：该项目利用启真医学知识库构建的中文医学指令数据集，并基于此在LLaMA-7B模型上进行指令精调，大幅提高了模型在中文医疗场景下效果，首先针对药品知识问答发布了评测数据集，后续计划优化疾病、手术、检验等方面的问答效果，并针对医患问答、病历自动生成等应用展开拓展。

- [【LLMs 入门实战 】 ChatMed 学习与实战]()
  - 论文名称：
  - 论文地址：https://github.com/michael-wzhu/ChatMed
  - 介绍：该项目推出ChatMed系列中文医疗大规模语言模型，模型主干为LlaMA-7b并采用LoRA微调，具体包括ChatMed-Consult : 基于中文医疗在线问诊数据集ChatMed_Consult_Dataset的50w+在线问诊+ChatGPT回复作为训练集；ChatMed-TCM : 基于中医药指令数据集ChatMed_TCM_Dataset，以开源的中医药知识图谱为基础，采用以实体为中心的自指令方法(entity-centric self-instruct)，调用ChatGPT得到2.6w+的围绕中医药的指令数据训练得到。

- [【LLMs 入门实战 】 XrayGLM 学习与实战]()
  - 论文名称：
  - 论文地址：https://github.com/WangRongsheng/XrayGLM
  - 介绍：该项目为促进中文领域医学多模态大模型的研究发展，发布了XrayGLM数据集及模型，其在医学影像诊断和多轮交互对话上显示出了非凡的潜力。

#### 第三重 法律领域

- [【LLMs 入门实战 】 LaWGPT 学习与实战]()：基于中文法律知识的大语言模型
  - 论文名称：
  - 论文地址：https://github.com/pengxiao-song/LaWGPT
  - 介绍：该系列模型在通用中文基座模型（如 Chinese-LLaMA、ChatGLM 等）的基础上扩充法律领域专有词表、大规模中文法律语料预训练，增强了大模型在法律领域的基础语义理解能力。在此基础上，构造法律领域对话问答数据集、中国司法考试数据集进行指令精调，提升了模型对法律内容的理解和执行能力。
- [【LLMs 入门实战 】 LexiLaw 学习与实战]()：中文法律大模型
  - 论文名称：
  - 论文地址：https://github.com/CSHaitao/LexiLaw
  - 介绍：LexiLaw 是一个基于 ChatGLM-6B微调的中文法律大模型，通过在法律领域的数据集上进行微调。该模型旨在为法律从业者、学生和普通用户提供准确、可靠的法律咨询服务，包括具体法律问题的咨询，还是对法律条款、案例解析、法规解读等方面的查询。
- [【LLMs 入门实战 】 Lawyer LLaMA 学习与实战]()：中文法律LLaMA
  - 论文名称：
  - 论文地址：https://github.com/AndrewZhe/lawyer-llama
  - 介绍：开源了一系列法律领域的指令微调数据和基于LLaMA训练的中文法律大模型的参数。Lawyer LLaMA 首先在大规模法律语料上进行了continual pretraining。在此基础上，借助ChatGPT收集了一批对中国国家统一法律职业资格考试客观题（以下简称法考）的分析和对法律咨询的回答，利用收集到的数据对模型进行指令微调，让模型习得将法律知识应用到具体场景中的能力。

#### 第四重 教育领域

- [【LLMs 入门实战 】 桃李（Taoli） 学习与实战]()
  - 论文名称：
  - 论文地址：https://github.com/blcuicall/taoli
  - 介绍：一个在国际中文教育领域数据上进行了额外训练的模型。项目基于目前国际中文教育领域流通的500余册国际中文教育教材与教辅书、汉语水平考试试题以及汉语学习者词典等，构建了国际中文教育资源库，构造了共计 88000 条的高质量国际中文教育问答数据集，并利用收集到的数据对模型进行指令微调，让模型习得将知识应用到具体场景中的能力。

#### 第五重 文化领域

- [【LLMs 入门实战 】 Firefly 学习与实战]()
  - 论文名称：
  - 论文地址：https://github.com/yangjianxin1/Firefly
  - 介绍：中文对话式大语言模型，构造了许多与中华文化相关的数据，以提升模型这方面的表现，如对联、作诗、文言文翻译、散文、金庸小说等。

#### 第六重 Coding

- [【LLMs 入门实战】 CodeGeeX2-6B 学习与实战](https://articles.zsxq.com/id_1bz3qf7r76wc.html)
  - DeepKE-LLM链接：
  - OpenKG地址：
  - Github 地址：https://github.com/THUDM/CodeGeeX2
  - 介绍：CodeGeeX2 是多语言代码生成模型 CodeGeeX 的第二代模型，基于 ChatGLM2 架构注入代码实现。得益于 ChatGLM2 的更优性能，CodeGeeX2-6B 在多项指标上取得了较大的性能提升。与 150 亿参数的 StarCoder-15B 相比，CodeGeeX2-6B 凭借 60 亿参数便具备了近 10% 的优势。

## 知识体系

> [关于 AIGC 那些你不知道的事](https://yaz1kaenukt.feishu.cn/mindnotes/OSsQbEhzomseronYdQmc6CmXnQH)
![](img/1.png)

> [关于 NLP 那些你不知道的事](https://yaz1kaenukt.feishu.cn/mindnotes/FOlQbPf6GmdzBQnBVsncnYGJnnd)
![](img/1.png)

## 加入学习群

> 扫码加小编，申请加入学习群
![](img/3.png)

## 参考

1. [Awesome-Chinese-LLM：收集和梳理中文LLM相关的开源模型、应用、数据集及教程等资料](https://mp.weixin.qq.com/s/Oy6XZNyN3kpsC6TfQYQb7A)


