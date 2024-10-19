from pydantic import BaseModel
import os
import sys
import torch
# 向量模型下载
from modelscope import snapshot_download
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from typing import Any, List, Optional
from langchain_community.vectorstores import Chroma
import os   
from typing import Any, List, Mapping, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from zhipuai import ZhipuAI
import embedding
from qa_chain import QA_chain_self

# app = FastAPI() # 创建 api 对象

prompt: str  # 用户 prompt
model: str = "glm-4-flash"  # 使用的模型
temperature: float = 0.1  # 温度系数
if_history: bool = False  # 是否使用历史对话功能
# API_Key
api_key: str = "652a160546149ef4e3ec0ff881beebfe.D3UaKuk7FmiUn9WQ"
# Secret_Key
secret_key: str = None
# access_token
access_token: str = None
# APPID
appid: str = None
# APISecret
Spark_api_secret: str = None
# Secret_key
Wenxin_secret_key: str = None
# 数据库路径
db_path: str = "./vector_db"
# 源文件路径
file_path: str = "./中国近现代史纲要：2023 年版(1).pdf"
# prompt template
prompt_template: str = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
问题: {question}
有用的回答:"""
# Template 变量
input_variables: list = ["context", "question"]
# Embdding
embedding: str = "m3e"
# Top K
top_k: int = 5
# embedding_key
embedding_key = "652a160546149ef4e3ec0ff881beebfe.D3UaKuk7FmiUn9WQ"

choiceproblem_file_path="./vector_db/choiceproblem"
analysis_vectordb_file_path="./vector_db/tao_8k"
embedding_path = "./AI-ModelScope/tao-8k"


def get_chain():
    return QA_chain_self(model=model, temperature=temperature, top_k=top_k,
                              choiceproblem_file_path=choiceproblem_file_path, analysis_vectordb_file_path=analysis_vectordb_file_path, persist_path=db_path,
                              api_key=api_key, embedding_path=embedding_path, template=prompt_template)



def get_response(message):
    return get_chain().answer(query=message)

