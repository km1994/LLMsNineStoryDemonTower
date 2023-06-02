# 【LLMs 入门实战 —— 十九】 聚宝盆(Cornucopia) 模型学习与实战
 
1. 代码：[jerry1993-tech/Cornucopia-LLaMA-Fin-Chinese](https://github.com/jerry1993-tech/Cornucopia-LLaMA-Fin-Chinese/tree/main)

## 一、前言

### 1.1 介绍

聚宝盆(Cornucopia) 开源了经过中文金融知识指令精调/指令微调(Instruct-tuning) 的LLaMA-7B模型。通过中文金融公开数据+爬取的金融数据构建指令数据集，并在此基础上对LLaMA进行了指令微调，提高了 LLaMA 在金融领域的问答效果。

### 1.2 软件资源

- CUDA 11.7
- Python 3.10
- pytorch 1.13.1+cu117

### 1.3 计算资源需求

目前训练设备为一张A100-SXM-80GB显卡，训练总轮次10轮。batch_size=64的情况下显存占用在40G左右、batch_size=96的情况下显存占用在65G左右。预计3090/4090显卡(24GB显存)以上显卡可以较好支持，根据显存大小来调整batch_size。

## 二、环境搭建

### 2.1 下载代码 

```s
    $ git clone https://github.com/jerry1993-tech/Cornucopia-LLaMA-Fin-Chinese.git
```

### 2.2 构建环境

```s
    $ conda create -n py310_chat python=3.10       # 创建新环境
    $ source activate py310_chat                   # 激活环境
```

### 2.3 安装依赖 

```s
    $ cd Cornucopia-LLaMA-Fin-Chinese
    $ pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.4 安装 lfs 方便本地下载 LLaMa 大模型

```s
    $ git lfs install

    # 下载7B模型到本地
    $ bash ./base_models/load.sh
```

> ./base_models/load.sh 源码介绍
```s
#!/bin/bash
# 下载 llama-7b-hf
base_model_pir="./base_models/llama-7b-hf"
if [ ! -d $base_model_pir ];then
  cd ../base_models/ || exit
  git clone https://huggingface.co/decapoda-research/llama-7b-hf
  cd ../ || exit
fi

# 下载 Linly-Chinese-LLaMA-7b-hf
base_model_pir="./base_models/Linly-Chinese-LLaMA-7b-hf"
if [ ! -d $base_model_pir ];then
  cd ../base_models/ || exit
  git clone https://huggingface.co/P01son/Linly-Chinese-LLaMA-7b-hf
  cd ../ || exit
fi

```

## 三、模型推理

### 3.1 模型下载

LoRA 权重可以通过 Huggingface 下载：

1. 对「decapoda-research/llama-7b-hf」进行指令微调的LoRA权重文件，下载后格式如下：

- 基于 12M 的中文金融知识问答 [HuggingFace](https://huggingface.co/yuyangmu125/lora-llama-fin-ori-fb)

```s
   lora-llama-fin-ori-fb/
       - adapter_config.json   # LoRA权重配置文件
       - adapter_model.bin     # LoRA权重文件
```

2. 对「Linly-AI/Chinese-LLaMA-7B」进行指令微调的LoRA权重文件

- 基于 14M 的中文金融知识问答 [HuggingFace](https://huggingface.co/yuyangmu125/lora-llama-fin-Linly-zh)

```s
   lora-llama-fin-Linly-zh/
       - adapter_config.json   # LoRA权重配置文件
       - adapter_model.bin     # LoRA权重文件
```

### 3.2 模型推理

目前在 ./instruction_data/infer.json 中提供了一些测试用例，也可替换成其它的数据集但注意格式保持一致

> ./instruction_data/infer.json 数据格式
```s
    {"instruction": "你好，我想了解一下创业板股票怎么买？", "input": "", "output": "您好，如果要购买创业板的股票，需要先开通创业板权限。\n开通创业板的三个条件：\n1.股东证券账户20个交易日日均资产10万以上；\n2.必须有24个月的交易经验；\n3.风险测试评级在积极型及以上；\n只要符合这三大条件的投资者都可以申请开通创业板交易权限。\n创业板是为中小型、创业性、科技型企业提供上市融资的场所。从这三类企业的角度出发，中小企业受限于规模，本身的抗风险能力就差；创业型企业的一般存在公司治理不完善的特点；对于科技型企业，高科技转化为产品或劳务具有不确定性，存在技术失败风险。所以对于普通投资者来说交易风险较大，把握不好还是要谨慎操作。"}
    ...
```

- 运行infer脚本

```s
    # 单模型推理 
    bash ./scripts/infer.sh  
```

```s
    # 多模型对比 
    bash ./scripts/comparison_test.sh
```

## 四、模型微调

### 4.1 微调数据集介绍

目前采用了公开和爬取的中文金融领域问答数据，涉及到保险、理财、股票、基金、贷款、信用卡、社保等。

指令微调的训练集数据示例如下：

```s
    问题：办理商业汇票应遵守哪些原则和规定？

    回答: 办理商业汇票应遵守下列原则和规定：1.使用商业汇票的单位，必须是在银行开立帐户的法人；2.商业汇票在同城和异地均可使用；3.签发商业汇票必须以合法的商品交易为基础；4.经承兑的商业汇票，可向银行贴现；5.商业汇票一律记名，允许背书转让；6.商业汇票的付款期限由交易双方商定，最长不得超过6个月；7.商业汇票经承兑后，承兑人即付款人负有到期无条件交付票款的责任；8.商业汇票由银行印制和发售。
```

### 4.2 微调数据集构建

若想用自己的数据集微调 LLaMA，请按照./instruction_data/fin_data.json的格式构建自己的数据集

```s
    {"instruction": "可转债风险大吗？和股票比风险大还是小？", "input": "", "output": "可转债和股票相比来说，风险是比较小一些的，打新债需要先开账户，开户完之后的2.个交易日就可以打新债了。\n开户流程如下\n1.打开优惠开户链接，点击开户；\n2.输入手机号和验证码；\n3.填写个人信息\n4.视频认证\n5.完成风险测评问题；\n6.3.方存管银行\n7.最后提交申请，等待审核。"}
    ...
```

### 4.3 模型 finetune

运行 finetune 脚本

```s
    $ bash ./scripts/finetune.sh
```

## 五、模型效果对比

| 测试输入 | 原始Llama输出 | Cornucopia(fin-ori-fb)输出                                                                                                                                                                                                                                                                                                  | Cornucopia(fin-Linly-zh)输出                                                                                                                                                                                                                                                                                                                           | 文心一言输出        | 讯飞星火认知输出 |
|:------------ |:----------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------|:------|
|老年人理财好还是存定期好？|老年人的理财办法应该适合于自身情况。如果他/她有足够的时间和投资能力，则建议他/她利用现在的时间与投资机会进行理财活动。因为老年人可以获得余额、保险、房地产等多种收入方式，所以他/她可以选择不同类型的理财项目，包括公开市场、非公开市场、基金、信托计划等。当然，老年人也应该注意其实需要的生活财务安全。| **若投资者风险承受能力比较高，那么可以选择理财产品进行投资，若投资者风险承受能力比较低，那么可以选择定期存款进行投资。目前理财产品和定期存款都是保本型产品，不会亏损本金，并且收益由投资标的决定，所以理财产品和定期存款相比较更加稳定。需要注意的是理财产品和定期存款没有同样的活期利率，而且理财产品每日计算收益，所以理财产品收益越高，投资者获得收益的概率也越高。9QDII基金是指中国市场上开放式外资基金，这类基金主要投资于：A股、港股、美股、英股、德股、日股、法股、台股、香港股、新加坡股等；QFII基金是指中国交易所开放式外资基金，这类基金主要投资于：A股、港股、美股、英股、德股、日股、法股、台股、香港股、新加坡股等。** | **对于风险承受能力比较低的老年人来说，选择定期存款比较好，因为其风险性比较低，收益也比较稳定，而对于风险承受能力比较高的老年人来说，选择理财产品比较好，因为其风险性比较大，收益也比较高。需要注意的是目前理财产品不是保本型产品，有亏损本金的可能，并且根据投资标的分为不同的风险等级，若投资者风险承受能力比较低，那么可以选择风险等级R2及以下的理财产品，若投资者风险承受能力比较高，那么可以选择风险等级R3及以上的理财产品。当前理财产品包括：债券型基金、股票型基金、混合型基金、指数基金和联接基金等，投资者可以根据自身实际情况选择合适的产品进行投资。**                                                                 |老年人存定期比较好。老年人没工作收入，资产总额也是在逐渐降低，而且老人容易生病，而医疗费支出较大。如果老年人投资理财产品，伴随着其产品净值的变化，是有可能会出现亏损现象的。而定期存款，其利率一般是固定的，期满返本还息，相对而言，安全系数高，盈利更持久。|作为AI语言模型，我没有个人观点和建议。但是，老年人理财和存定期都有其优缺点。老年人理财的优点是可以更好地规划自己的财务状况，包括退休金、医疗保险、子女教育等方面的支出。此外，老年人可以通过投资股票、基金等金融产品来增加财富。存定期的优点是风险较小，收益相对稳定。老年人可以选择不同期限的定期存款，根据自己的需求和风险承受能力来选择。综合来看，老年人应该根据自己的实际情况和需求来选择适合自己的理财方式。如果老年人有足够的资金和风险承受能力，可以考虑投资股票、基金等金融产品来增加财富。如果老年人更注重稳定性和安全性，可以选择存定期来保障自己的财务状况。|
|股票和基金能当天随买随卖吗？|Yes, stocks and funds can be bought and sold on the same day.| **不能，股票和基金实行T+1交易，交易日当天买入，按照买入当天收盘时的净值计算，第二个交易日确认份额，份额确认后计算收益，所以在交易日买入就要等到第二个交易日才能获得收益。比如投资者周一买入了某只股票，周二市场上公布的净值为10元，那么投资者可以根据这些数据计算收益：1000×10=10000；10000/10=1000。若投资者持有此股票超过30个工作日，那么可以直接转换成现金，也就是将该股票全部或者部分提前支取。需要注意的是，目前没有任何法律法规限制投资者在交易日买入基金、股票，因此投资者在交易日买入基金、股票都能进行当天开放型交易。同时，投资者在交易日买入股票之后，其股票会自动转换为现金，并且投资者在交易日买入股票之后，其股票会自动转换为涨跌幅限制（20％左右），从而影响投资者的手续费和收益。**| **不能，股票交易时间：周一至周五上午9:30-11:30，下午13:00-15:00，法定节假日不交易。基金交易时间：周一至周五上午9:30-11:30，下午13:00-15:00，法定节假日不交易。股票实行T+1交易，按照市场实时价格进行成交，遵循价格优先、时间优先的原则，沪深两市股票涨跌幅限制为10%，创业板和科创板股票涨跌幅限制为20%，北交所涨跌幅限制为30%，ST以及*ST涨跌幅限制为5%（创业板和科创板涨跌幅还没有这个限制），港股涨跌幅限制为20%，新三板涨跌幅限制为10%（创业板和科创板涨跌幅还没有这个限制），对于投资者来说，股票交易比较灵活，可以根据自身情况进行交易，并且股票交易手续费比较低，基金交易手续费也比较低，同时股票波动比较大，风险也比较大。** |不可以，股票基金实行T+1交易，交易日当天买入（交易时间），按照买入当天收盘时的净值计算，第二个交易日确认份额，份额确认后才能将基金卖出，一般交易日当天买入，第三个交易日才能卖出，第三个交易日随时可以卖出。| 股票和基金的交易方式因不同的证券市场而异。在一些证券市场上，股票和基金是可以当天随买随卖的，也就是所谓的“T+0”交易。但是，在另一些证券市场上，股票和基金是不能当天随买随卖的，需要等待一定的交易时间后才能进行买卖。在中国A股市场中，股票和基金是不能当天随买随卖的，需要进行T+1或T+2的交易。这意味着如果您今天买入了一只股票或基金，那么您需要等到明天才能卖出它。 在其他一些国家和地区，股票和基金的交易方式可能会有所不同。因此，如果您想进行股票或基金的投资，建议您了解当地的交易规则和市场情况。                           |


## 填坑笔记


## 致谢

本项目参考了以下开源项目，在此对相关项目和研究开发人员表示感谢。

- Facebook LLaMA: https://github.com/facebookresearch/llama
- Stanford Alpaca: https://github.com/tatsu-lab/stanford_alpaca
- alpaca-lora by @tloen: https://github.com/tloen/alpaca-lora
- Huatuo-Llama-Med-Chinese: https://github.com/SCIR-HI/Hatuo-Llama-Med-Chinese
- 文心一言 https://yiyan.baidu.com/welcome 本项目的logo由文心一言自动生成
- 讯飞星火认知 https://xinghuo.xfyun.cn/desk

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

## 参考

1. [聚宝盆(Cornucopia): 基于中文金融知识的LLaMA模型，效果媲美「度小满轩辕」](https://zhuanlan.zhihu.com/p/633736418)
2. 代码：[jerry1993-tech/Cornucopia-LLaMA-Fin-Chinese](https://github.com/jerry1993-tech/Cornucopia-LLaMA-Fin-Chinese/tree/main)

