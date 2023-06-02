# LLMsNineStoryDemonTower LLMs九层妖塔
【LLMs九层妖塔】分享一下打怪(ChatGLM、Chinese-LLaMA-Alpaca、MiniGPT-4、FastChat、LLaMA、gpt4all等)实战与经验，

![LLMs九层妖塔 视频介绍](https://github.com/km1994/LLMsNineStoryDemonTower/blob/main/mp4/LLMs九层妖塔挑战赛.mp4)
> [LLMs九层妖塔 视频介绍 地址](https://github.com/km1994/LLMsNineStoryDemonTower/blob/main/mp4/LLMs九层妖塔挑战赛.mp4)

- [LLMsNineStoryDemonTower LLMs九层妖塔](#llmsninestorydemontower-llms九层妖塔)
  - [【LLMs 入门实战系列】](#llms-入门实战系列)
    - [第一层 ChatGLM-6B](#第一层-chatglm-6b)
    - [第二层 Stanford Alpaca 7B](#第二层-stanford-alpaca-7b)
    - [第三层 Chinese-LLaMA-Alpaca](#第三层-chinese-llama-alpaca)
    - [第四层 小羊驼 Vicuna](#第四层-小羊驼-vicuna)
    - [第五层 MiniGPT-4](#第五层-minigpt-4)
    - [第六层 GPT4ALL](#第六层-gpt4all)
    - [第七层 AutoGPT](#第七层-autogpt)
    - [第八层 MOSS](#第八层-moss)
    - [第九层 BLOOM](#第九层-bloom)
    - [第十层 BELLE](#第十层-belle)
    - [第十一层 LLMTune](#第十一层-llmtune)
<<<<<<< HEAD
    - [第十二层 VisualGLM-6B](#第十二层-visualglm-6b)
    - [第十三层 ChatRWKV](#第十三层-chatrwkv)
    - [第十四层 聚宝盆(Cornucopia)](#第十四层-聚宝盆cornucopia)
    - [第十五层 Guanaco](#第十五层-guanaco)
    - [第十六层 Massively Multilingual Speech (MMS，大规模多语种语音)](#第十六层-massively-multilingual-speech-mms大规模多语种语音)
=======
>>>>>>> 66d9c9c237c1f29ad4ea8ccce513de30e9d1bbe6
  - [学习群](#学习群)
  - [优秀笔记](#优秀笔记)
    - [第一层](#第一层)
    - [优秀笔记](#优秀笔记-1)
  - [参考](#参考)

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

### 第九层 BLOOM

- [【LLMs 入门实战 —— 十四 】 BLOOM 模型学习与实战](https://articles.zsxq.com/id_wd97899pkjqj.html)
  - 介绍：大型语言模型（LLMs）已被证明能够根据一些演示或自然语言指令执行新的任务。虽然这些能力已经导致了广泛的采用，但大多数LLM是由资源丰富的组织开发的，而且经常不对公众开放。作为使这一强大技术民主化的一步，我们提出了BLOOM，一个176B参数的开放性语言模型，它的设计和建立要感谢数百名研究人员的合作。BLOOM是一个仅有解码器的Transformer语言模型，它是在ROOTS语料库上训练出来的，该数据集包括46种自然语言和13种编程语言（共59种）的数百个来源。我们发现，BLOOM在各种基准上取得了有竞争力的性能，在经历了多任务提示的微调后，其结果更加强大。
  - 模型地址：https://huggingface.co/bigscience/bloomz

### 第十层 BELLE

- [【LLMs 入门实战 —— 十五 】 BELLE 模型学习与实战](https://articles.zsxq.com/id_gxebzsadfpr2.html)
  - 介绍：相比如何做好大语言模型的预训练，BELLE更关注如何在开源预训练大语言模型的基础上，帮助每一个人都能够得到一个属于自己的、效果尽可能好的具有指令表现能力的语言模型，降低大语言模型、特别是中文大语言模型的研究和应用门槛。为此，BELLE项目会持续开放指令训练数据、相关模型、训练代码、应用场景等，也会持续评估不同训练数据、训练算法等对模型表现的影响。BELLE针对中文做了优化，模型调优仅使用由ChatGPT生产的数据（不包含任何其他数据）。
  - github 地址: https://github.com/LianjiaTech/BELLE

### 第十一层 LLMTune

- [【LLMs 入门实战 —— 十六 】 LLMTune 模型学习与实战](https://articles.zsxq.com/id_1hg51c292bu6.html)
  - 动机：大语言模型虽然能力很强，目前开源生态也很丰富，但是在特定领域微调大模型依然需要大规格的显卡。例如，清华大学发布的ChatGLM-6B，参数规模60亿，在没有量化的情况下微调需要14GB显存（parameter-efficient fine-tuning，PEFT)。在没有任何优化的前提下，每10亿参数的全精度（32bit）模型载入到显存中就需要4GB，而int8量化后也需要1GB显存。而目前开源最强的模型LLaMA，其最高参数维650亿规模，全精度模型载入就需要260GB，显然已经超出了大部分人的硬件水平。更不要说对模型进行微调（微调需要训练更新参数，推理只需要前向计算即可，因此，微调需要更多的显存才能支持）。
  - 介绍：Cornell Tech开源的LLMTune就是为了降低大模型微调难度所提出的一种解决方案。对于650亿参数的LLaMA模型微调仅需要40GB显存即可。
  - github 地址: https://github.com/kuleshov-group/llmtune

<<<<<<< HEAD
### 第十二层 VisualGLM-6B

- [【LLMs 入门实战 —— 十七 】 VisualGLM-6B 模型学习与实战](https://articles.zsxq.com/id_4pzgwnwl2zjc.html)
  - 动机：OpenAI 的GPT-4样例中展现出令人印象深刻的多模态理解能力，但是能理解图像的中文开源对话模型仍是空白。
  - 介绍：VisualGLM-6B 是一个开源的，支持图像、中文和英文的多模态对话语言模型，语言模型基于 ChatGLM-6B，具有 62 亿参数；图像部分通过训练 BLIP2-Qformer 构建起视觉模型与语言模型的桥梁，整体模型共 78 亿参数。

VisualGLM-6B 依靠来自于 CogView 数据集的30M高质量中文图文对，与 300M 经过筛选的英文图文对进行预训练，中英文权重相同。该训练方式较好地将视觉信息对齐到 ChatGLM 的语义空间；之后的微调阶段，模型在长视觉问答数据上训练，以生成符合人类偏好的答案。
  - github 地址:https://github.com/THUDM/VisualGLM-6B

### 第十三层 ChatRWKV

- [【LLMs 入门实战 —— 十八 】 ChatRWKV 模型学习与实战]()
  - 目前 RWKV 有大量模型，对应各种场景，各种语言，请选择合适的模型：
    - Raven 模型：适合直接聊天，适合 +i 指令。有很多种语言的版本，看清楚用哪个。适合聊天、完成任务、写代码。可以作为任务去写文稿、大纲、故事、诗歌等等，但文笔不如 testNovel 系列模型。
    - Novel-ChnEng 模型：中英文小说模型，可以用 +gen 生成世界设定（如果会写 prompt，可以控制下文剧情和人物），可以写科幻奇幻。不适合聊天，不适合 +i 指令。
    - Novel-Chn 模型：纯中文网文模型，只能用 +gen 续写网文（不能生成世界设定等等），但是写网文写得更好（也更小白文，适合写男频女频）。不适合聊天，不适合 +i 指令。
    - Novel-ChnEng-ChnPro 模型：将 Novel-ChnEng 在高质量作品微调（名著，科幻，奇幻，古典，翻译，等等）。
  - github: https://github.com/BlinkDL/ChatRWKV
  -  模型文件：https://huggingface.co/BlinkDL

### 第十四层 聚宝盆(Cornucopia) 

- [【LLMs 入门实战 —— 十九】 聚宝盆(Cornucopia) 模型学习与实战](https://github.com/km1994/LLMsNineStoryDemonTower/tree/main/Cornucopia_19)
  - 聚宝盆(Cornucopia) 开源了经过中文金融知识指令精调/指令微调(Instruct-tuning) 的LLaMA-7B模型。通过中文金融公开数据+爬取的金融数据构建指令数据集，并在此基础上对LLaMA进行了指令微调，提高了 LLaMA 在金融领域的问答效果。
  - github: [jerry1993-tech/Cornucopia-LLaMA-Fin-Chinese](https://github.com/jerry1993-tech/Cornucopia-LLaMA-Fin-Chinese/tree/main)

### 第十五层 Guanaco

- [【LLMs 入门实战 —— 二十 】 Guanaco 模型学习与实战](https://github.com/km1994/LLMsNineStoryDemonTower/tree/main/Guanaco_20)
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

### 第十六层 Massively Multilingual Speech (MMS，大规模多语种语音)

- [【LLMs 入门实战 —— 二十 】 Massively Multilingual Speech (MMS，大规模多语种语音) 模型学习与实战](https://github.com/km1994/LLMsNineStoryDemonTower/tree/main/speech_MMS_21)
  - 论文：[Scaling Speech Technology to 1,000+ Languages](https://research.facebook.com/publications/scaling-speech-technology-to-1000-languages/)
  - 代码：[fairseq/tree/main/examples/mms](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)
  - 公告：https://ai.facebook.com/blog/multilingual-model-speech-recognition/
  - 介绍：Meta 在 GitHub 上再次开源了一款全新的 AI 语言模型——Massively Multilingual Speech (MMS，大规模多语种语音)，它与 ChatGPT 有着很大的不同，这款新的语言模型可以识别 4000 多种口头语言并生成 1100 多种语音（文本到语音）。
=======
>>>>>>> 66d9c9c237c1f29ad4ea8ccce513de30e9d1bbe6

## 学习群

![[学习群二维码](img/20230516092740.jpg)](img/20230516092740.jpg)
> 二维码如果过期，可以加 wx: yzyykm666 加群

## 优秀笔记

### 第一层

### 优秀笔记

1. [杨夕](https://mp.weixin.qq.com/s/4QNgF6nAUo8imSaIB_OWmg)
2. [奔腾](https://articles.zsxq.com/id_k2qzsps7zw21.html)
3. [逸尘](https://articles.zsxq.com/id_zzfqt88sw4rl.html)
4. [此方一泉](https://t.zsxq.com/0dEp8PDcW)
5. [vezel](http://t.csdn.cn/hWn9D)
6. [徐生](https://zhuanlan.zhihu.com/p/627358709)
7. [多点微笑](https://articles.zsxq.com/id_velwvtmfhrwz.html)
8. [小固](https://zhuanlan.zhihu.com/p/627333187)
9. [土狼](https://zhuanlan.zhihu.com/p/627358709)
10. [0](https://github.com/Wesley12138/LLM)
11. [Welch](https://t.zsxq.com/0dJhaaGRW)
12. [九猫](https://articles.zsxq.com/id_7g0g65fbsluo.html)


## 参考

1. [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
2. [Stanford Alpaca 7B](https://github.com/tatsu-lab/stanford_alpaca)
3. [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
4. [Vicuna](https://github.com/lm-sys/FastChat)
5. [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
6. [GPT4ALL](https://github.com/nomic-ai/gpt4all)
7. [Auto-GPT](hhttps://github.com/Significant-Gravitas/Auto-GPT)
8. [MOSS](https://github.com/OpenLMLab/MOSS/tree/main)
