import gradio as gr
import random
import time
from api import get_chain
from modelscope import snapshot_download
from embedding import create_db
def respond(message, chat_history):
    chain = get_chain()
    bot_message = chain.answer(query=message)['text']
    # bot_message = message
    
    chat_history.append((message, bot_message))
    time.sleep(2)
    return "", chat_history

with gr.Blocks(title="党员问答助手") as demo:
    with gr.Column(variant="default", elem_classes="outer"):
      with gr.Column(elem_classes="title"):
          gr.Markdown("# <center>党员问答助手 </center>")
    # 向量模型下载
    snapshot_download("AI-ModelScope/tao-8k", cache_dir='./')
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    demo.launch(root_path="/dsw-696320/proxy/7860/")