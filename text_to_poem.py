# 加载int4量化后的模型
# from lmdeploy import turbomind as tm
# class TextToPoem(object):
#     def __init__(self):
#         # load model
#         self.model_path = r"E:\\my_model\\model_quant"  # "/root/gushi2/merged"
#         self.tm_model = tm.TurboMind.from_pretrained(self.model_path, model_name='internlm-chat-7b')
#         self.generator = self.tm_model.create_instance()
#
#     def text_to_poem(self, questions):
#         # query = input()   #"根据明月这个关键词给我生成一个古诗"
#         prompt = self.tm_model.model.get_prompt(f"{questions}")
#         input_ids = self.tm_model.tokenizer.encode(prompt)
#         # inference
#         for outputs in self.generator.stream_infer(session_id=0,input_ids=[input_ids]):
#             res, tokens = outputs[0]
#             # print(outputs[0])
#             # res = outputs[0]
#         response = self.tm_model.tokenizer.decode(res.tolist())
#         return response
#
# if __name__ == '__main__':
#     posts = TextToPoem()
#     for i in range(3):
#         my_input = input("请输入关键词或句子：")
#         poem = posts.text_to_poem(questions=my_input)
#         print(poem)


# 正常的加载merged模型
import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM


class TextToPoem(object):
    def __init__(self):
        self.model_dir = r"HOOK123/poem-soul"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, device_map="auto", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, device_map="auto",  trust_remote_code=True, torch_dtype=torch.float16,  offload_folder=r"E:\my_model\merged\temp")

    def text_to_poem(self,questions):
        model = self.model.eval()
        response, history = model.chat(self.tokenizer, f"{questions}", history=[])
        answer = response
        return answer


if __name__ == '__main__':
    posts = TextToPoem()
    for i in range(3):
        poem = posts.text_to_poem(questions="根据孤影这个关键词给我生成一首古诗")
        print(poem)