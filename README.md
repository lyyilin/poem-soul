# poem-soul
<!-- PROJECT SHIELDS -->
<br />
<!-- PROJECT LOGO -->


<p align="center">
  <a href="[https://github.com/lyyilin/poem-soul]">
    <img src="imgs/robot.png" alt="Logo" width="20%">
  </a>

<h3 align="center">诗魂</h3>
  <p align="center">
    <br />
    <a href="https://github.com/lyyilin/poem-soul"><strong>探索本项目的文档</strong></a>
    <br />
    <br />
    <a href="https://openxlab.org.cn/apps/detail/HOOK/poem-soul">查看Demo</a>
  </p>

</p>

## 本项目是基于书生·浦语大模型实战营，产出的作品
**模型名称**：诗魂-古诗生成

**模型概览**：

诗魂是一个基于深度学习的模型，旨在从现代语言中生成具有古诗韵味的文本。
该模型通过对大量古诗进行学习，掌握了古诗的韵律、意象和语言风格，从而能够生成与古诗相媲美的文本。

**用途与功能**：

1. **韵律分析**：通过对古诗的韵律进行分析，模型能够生成符合古诗韵律的现代文。
2. **意象再现**：模型能够学习古诗中的意象，并在生成文本时再现这些意象，使现代文具有古诗的意境之美。
3. **风格模仿**：通过模仿古诗的语言风格，模型生成的文本在表达上具有古风之美。
4. **自动续写**：用户提供一句古诗，模型能够自动续写，形成完整的诗篇。
5. **主题创作**：用户指定主题或关键词，模型能够创作出与主题相关的古诗韵味文本。

**使用情境**：

适用于对古诗文化感兴趣的用户，可以帮助他们快速生成具有古诗韵味的文本，用于创作、文学研究、诗歌欣赏等方面。

## 模型使用 Usage
```python
import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "HOOK123/poem-soul"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model = model.eval()

system_prompt = """你是一位专业的古诗词专家，根据用户的输入给出对应的古诗"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("User  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break
    response, history = model.chat(tokenizer, input_text, history=messages)
    messages.append((input_text, response))
    print(f"robot >>> {response}")
```

**训练方法**：

* **数据来源**：
* **训练方法**：通过对internlm-chat-7b使用xtuner微调训练得到

**数据来源**：
* **数据来源**：清华团队-九歌数据集
![数据来源](https://img-blog.csdnimg.cn/direct/c4c2a6dcfff44dfbaecedbf4fe585a0e.png)

## 特别鸣谢

- [辉哥]([https://github.com/sanbuphy](https://github.com/zhanghui-china))
- [上海人工智能实验室](https://www.shlab.org.cn/)
- 感谢浦语提供的大量算力！！！

**优势与局限**：

* **优势**：通过用户给出的情景或者意向词能够快速生成具有古诗韵味的文本，为用户提供丰富的创作素材。
* **局限**：对于特定主题或风格的古诗，可能无法完全模仿其精髓，偶尔会有小bug。

## 组内成员
- 李一林
- 邓雅文
- 邓超
- Leo
