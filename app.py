import gradio as gr
from trans import translate
from PIL import Image
from image_to_txt import image_to_txt
from text_to_poem import TextToPoem



# 图片描述
# Image inference
def get_caption(inputs):
    inputs = Image.fromarray(inputs)
    my_class = image_to_txt()
    my_trans = translate()
    text = my_class.to_txt(inputs)
    chinese_reslut = my_trans.english_to_chinese(text[0])  # 百度翻译调用
    return text[0], chinese_reslut


# internlm生成古诗
posts = TextToPoem()
def interlm(outputs, choice):
    poem = posts.text_to_poem(questions=f"{outputs}")
    return poem


# 三个按钮生成不同的格式
def change_textbox(outputs, choice):
    if choice == "五言绝句":
        return interlm(outputs, choice)

    elif choice == "五言律诗":
        return interlm(outputs, choice)
    # elif choice == "七言律诗":
    #     return wenxin(outputs, choice)

def tts():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Row(equal_height=False):
            with gr.Column(variant='panel'):
                inputs = gr.Image(label="需要生成的描述 ")
                text_button_img = gr.Button("确定上传")
                outputs_en = gr.Textbox(label='生成后的英文结果为')
                Englist_output = gr.Textbox(label='生成的中文结果',  show_copy_button=True)
            text_button_img.click(fn=get_caption, inputs=inputs, outputs=[outputs_en, Englist_output])
        with gr.Row(equal_height=False):
            with gr.Row(variant='panel'):
                user_inputs = gr.Textbox(label="输入您的关键词或复制上述输出结果用以生成您想要的古诗")
                user_button = gr.Button("确定上传")
                radio = gr.Radio(["五言绝句", "五言律诗"], label="请选择你想生成古诗的格式")
                text = gr.Textbox( interactive=True, show_copy_button=True)
        radio.change(fn=change_textbox, inputs=[user_inputs, radio], outputs=text)
        user_button.click(fn=interlm, inputs=[user_inputs, radio], outputs=text)

    return demo


if __name__ == '__main__':

    with gr.Blocks(css='style.css') as demo:
        gr.Markdown(
            "# <center> \N{fire} 图生文 </center>")
        with gr.Tabs():
            with gr.TabItem('\N{clapper board} 描述图片'):
                with gr.Column():
                    tts()
    demo.launch()