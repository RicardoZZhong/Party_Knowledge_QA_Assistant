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
from embedding import get_embedding, get_vectordb, load_knowledge_db
# 继承自 langchain_core.language_models.llms.LLM,用于调用 ZhipuAI 的语言模型服务
class ZhipuAILLM(LLM):
    # 默认选用 glm-4 模型
    model: str = "glm-4"
    # 温度系数
    temperature: float = 0.1
    # API_Key
    api_key: str = "652a160546149ef4e3ec0ff881beebfe.D3UaKuk7FmiUn9WQ"
    max_tokens: int = 2048

    # 定义 _call 方法：
    # 这个方法实现了实际的 API 调用逻辑：
    # 初始化 ZhipuAI 客户端。
    # 生成请求参数messages。
    # 调用 chat.completions.create 方法获取响应。
    # 返回响应中的内容，如果没有结果则返回错误信息。
    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        # 生成 GLM 模型请求参数的方法：
        # 生成 GLM 模型的请求参数 messages，包括系统消息和用户输入
        def gen_glm_params(prompt):
            '''
            构造 GLM 模型请求参数 messages
            请求参数：
                prompt: 对应的用户提示词
            '''
            messages = [{"role": "user", "content": prompt}]
            return messages

        client = ZhipuAI(
            api_key=self.api_key
        )

        messages = gen_glm_params(prompt)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        if len(response.choices) > 0:
            return response.choices[0].message.content
        return "generate answer error"

    # 定义属性方法：
    # _default_params：返回调用 API 的默认参数。
    # _llm_type：返回模型类型的字符串标识。
    # _identifying_params：返回模型的标识参数。
    # 首先定义一个返回默认参数的方法
    @property
    def _default_params(self) -> Dict[str, Any]:
        normal_params = {
            "temperature": self.temperature,
        }
        # print(type(self.model_kwargs))
        return {**normal_params}

    @property
    def _llm_type(self) -> str:
        return "Zhipu"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}