import json
import random
# 输入文件路径
input_file = r"C:\Users\14475\Desktop\ccpc_train_v1.0.json"
# 输出文件前缀
output_prefix = 'tran_dataset_'
# 每个输出文件的记录数
records_per_file = 100000

# 记录索引的起始值
start_index = 0


with open(input_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()

    # 初始化索引变量
    row_num = 0
    conversations = []

    # 遍历每一行JSON数据
    for line in lines:
        data = json.loads(line)  # 将JSON字符串转换为Python对象
        keywords = data['keywords']  # 获取keywords字段的值
        content = data['content'].split("|")  # 获取content字段的值
        # author = data['author']  # 获取author字段的值（这里我们假设author字段包含所需的信息）
        system_message = f"你是一个专业的古诗歌专家，你知道很多古诗。用户报上关键词后，你可以把包含关键词的古诗告诉用户"  # 定义系统消息
        if len(keywords[0:].split(" ")) == 4:
            shuzi = random.randint(0,3)
            input_message = keywords.split(" ")[shuzi]
        elif len(keywords[0:].split(" ")) == 3:# 假设我们使用keywords中的第一个作为输入消息（根据您的需求进行修改）
            shuzi2 = random.randint(0, 2)
            input_message = keywords.split(" ")[shuzi2]
        else:
            input_message = keywords.split(" ")[0]  # 假设我们使用keywords中的第一个作为输入消息（根据您的需求进行修改）

        output_message = f"生成的古诗为:\n{content[0]},\n{content[1]}。\n{content[2]},\n{content[3]}。"   # 生成输出消息
        # print("根据" + input_message + "这个关键词写一首古诗")
        input_message = "根据" + input_message + "这个关键词写一首古诗"
        new_record = {
            "conversation": [
                {
                    "system": system_message,
                    "input": input_message,
                    "output": output_message
                }
            ]
        }
        conversations.append(new_record)  # 将记录添加到conversations列表中
        row_num += 1  # 更新行号以写入JSON文件
        if row_num % records_per_file == 0:  # 当达到每个文件的记录数时，创建一个新的输出文件并写入数据
            with open(f'{output_prefix}{start_index}.json', 'w',encoding='utf-8') as file:
                json.dump(conversations, file, ensure_ascii=False, indent=4)  # 使用indent参数来格式化输出JSON字符串，并将ensure_ascii设置为False，这样就可以直接写入中文字符而不是转义序列了。
            start_index += 1  # 更新索引值以创建下一个输出文件
            conversations = []  # 重置conversations列表以准备写入下一个文件