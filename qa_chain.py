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
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
# 导入PromptTemplate类，用于创建问题模板
from langchain.chains import RetrievalQA
# 导入RetrievalQA类，用于执行检索问答
import sys
# 导入sys模块，用于操作系统相关功能
from langchain_community.vectorstores import Chroma
from zhipuai import ZhipuAI
import llm
from embedding import get_embedding, get_vectordb, load_knowledge_db
from llm import ZhipuAILLM

sys.path.append("../")
# 添加上级目录到模块搜索路径
# from qa_chain.model_to_llm import model_to_llm
# # 导入model_to_llm函数，用于将模型转换为大语言模型(LLM)
# from qa_chain.get_vectordb import get_vectordb
# # 导入get_vectordb函数，用于获取向量数据库

# 定义 QA_chain_self 类：
# 该类用于创建和管理一个问答链系统，不带历史记录，类的描述文档说明了各个参数的用途。
class QA_chain_self():
    # 类描述
    """
    不带历史记录的问答链
    - model：调用的模型名称
    - temperature：温度系数，控制生成的随机性
    - top_k：返回检索的前k个相似文档
    - file_path：建库文件所在路径
    - persist_path：向量数据库持久化路径
    - api_key：所有模型都需要
    - embeddings：使用的embedding模型
    - embedding_key：使用的embedding模型的秘钥（智谱或者OpenAI）
    - template：可以自定义提示模板，没有输入则使用默认的提示模板default_template_rq
    """

    # 默认的提示模板，用于构建问答的输入
    default_template_rq = """
    使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    有用的回答:"""

    template1 = """给你一个任务：给定一段文本，这段文本中第一段是题目，题目中会有括号，后面四行为A，B，C，D四个选项，最后一行是题目的答案。你需要做的是：将题目中的括号以及括号内的内容替换为答案并输出问题，注意不需要输出选项。
    不需要你检查指出题目的对错，不要输出除了以上任务之外的任何回答。
    这是你需要处理的文本：{question}
    """
    template2 = """使用以下背景知识来回答最后的问题。不要试图编造答案。尽量简明扼要地回答。
    背景知识：{context}
    问题：{query}"""


    # 构造函数，初始化类实例
    # 初始化向量数据库和LLM
    # 初始化 PromptTemplate 和 RetrievalQA 类
    def __init__(self, model: str, temperature: float = 0.0, top_k: int = 4, choiceproblem_file_path: str = None,
                 analysis_vectordb_file_path: str = None, persist_path: str = None, api_key: str = None,
                 embedding="zhipuai", embedding_path=None, template=default_template_rq):
        # 类属性初始化
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        # self.file_path = file_path
        self.persist_path = persist_path
        self.api_key = api_key
        self.embedding = embedding
        # self.embedding_key = embedding_key
        self.embedding_path = embedding_path
        self.template = template
        client = ZhipuAI(api_key="652a160546149ef4e3ec0ff881beebfe.D3UaKuk7FmiUn9WQ")
        # 加载已存在的向量数据库
        # self.choiceproblem_vectordb = get_vectordb(choiceproblem_file_path, persist_path + "/choiceproblem", embedding_path)
        self.analysis_vectordb = get_vectordb(analysis_vectordb_file_path, persist_path + "/tao_8k", embedding_path)
        self.llm = ZhipuAILLM()

        # # 初始化PromptTemplate和RetrievalQA类
        # self.QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=self.template)
        # self.retriever = self.vectordb.as_retriever(search_type="similarity", search_kwargs={'k': self.top_k})
        # self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.retriever,
        #                                             return_source_documents=True,
        #                                             chain_type_kwargs={"prompt": self.QA_CHAIN_PROMPT})
        self.prompt1 = PromptTemplate(
            input_variables=["question"],
            template=self.template1
        )
        self.chain1 = LLMChain(
            llm=self.llm, prompt=self.prompt1
        )

        self.prompt2 = PromptTemplate(
            input_variables=["context", "query"],
            template=self.template2
        )
        self.chain2 = LLMChain(
            llm=self.llm, prompt=self.prompt2
        )


    # 提供问答功能的方法，根据用户问题调用问答链并返回结果
    def answer(self, query: str = None, temperature=None, top_k=4):
        """
        核心方法，调用问答链
        arguments:
        - question：用户提问
        """
        # 检查问题是否为空
        if len(query) == 0:
            return ""

        # 如果没有指定温度或top_k参数，则使用初始化时的值
        if temperature is None:
            temperature = self.temperature
        if top_k is None:
            top_k = self.top_k

        # sim_docs = self.choiceproblem_vectordb.similarity_search(query, k=1)
        # question = ""
        # for sim_doc in sim_docs:
        #     print(sim_doc.page_content)
        #     print("--------------")
        #     question = question + sim_doc.page_content+'\n'
        # response = self.chain1(question)
        # print(response)
        # print("---------\n")
        # context = response['text']
        context = ""
        related_analysis = self.analysis_vectordb.similarity_search(query, k=3)
        for related_doc in related_analysis:
            # print(type(related_doc))
            context = context + related_doc.page_content
        result = self.chain2({'context': context, 'query': query})
        # 调用问答链并返回结果
        # result = self.qa_chain({"query": question, "temperature": temperature, "top_k": top_k})
        return result