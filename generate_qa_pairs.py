from typing import List

from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.evaluation import BinaryClassificationEvaluator, RerankingEvaluator
from zhipuai import ZhipuAI
from sentence_transformers import losses
import re
import json
from langchain_core.documents import Document
from tqdm import tqdm
from torch.utils.data import DataLoader
class QaPairs():
    '''存储List[dict]类型数据'''

    def __init__(self, qa_pairs: List[dict]):
        self.qa_pairs = qa_pairs
        

    def save_json(self, path: str):
        '''将数据存储为json格式'''

        with open(path, "w", encoding='utf-8') as f:
            json.dump(self.qa_pairs, f, ensure_ascii=False, indent=4)

    @classmethod
    def from_json(cls, path:str) -> 'QaPairs':
        '''读取json格式数据'''

        with open(path) as f:
            data = json.load(f)
        return cls(data)


llm_list = ['glm-4-plus', 'glm-4-flash', 'glm-4', 'glm-4v', 'glm-3-turbo', 'gpt-3.5-turbo']

PROMPT = '''
下面是上下文信息。 
 
--------------------- 
{context_str} 
--------------------- 
 
给定上下文信息，没有先验知识。 
仅根据下面的查询生成问题。 
 
你是一位老师/教授。你的任务是为即将到来的
测验/考试设置{num_questions_per_page}个问题以及问题涉及到的原文内容
在整个文件中，问题的性质应该是多样化的。
将问题限制在提供的上下文信息之内。
按照一下格式输出：
问题1：
问题

原文内容1：
内容

问题2：
问题

原文内容2：
内容
'''

def list_generate_qa_pairs(
        texts: List[str],
        num_questions_per_page: int = 2,
        model: str = 'glm-4-plus',
) -> QaPairs:
    '''借助大模型从给定的texts里提取出问题与对应的答案'''

    if model not in llm_list:
        raise ValueError('你选择的模型暂时不被支持'
                            '''请使用'glm-4', 'glm-4v', 'glm-3-turbo' 中的一个作为model的参数''')
    elif model in llm_list[:3]:
        llm = ZhipuAI(
            api_key="652a160546149ef4e3ec0ff881beebfe.D3UaKuk7FmiUn9WQ"
        )
    qa_pairs = []

    for text in tqdm(texts):
        if len(text) > 200:
            prompt = PROMPT.format(
                context_str=text,
                num_questions_per_page=num_questions_per_page
            )
            response = llm.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            matches = re.findall(
                r'问题\d+：(.*?)原文内容\d+：(.*?)((?=问题\d+：)|$)',
                response.choices[0].message.content,
                re.DOTALL
            )
            for _, match in enumerate(matches):
                qa = {
                    'query': match[0].strip(),
                    'answer': match[1].strip()
                }
                qa_pairs.append(qa)
    return QaPairs(qa_pairs=qa_pairs)

def docs_generate_qa_pairs(
        docs: List[Document], 
        num_questions_per_page: int = 2,
        model: str = 'glm-4-plus'
) -> QaPairs:
    '''借助大模型从给定的docs里提取出问题与对应的答案'''
    list_doc = [doc.page_content for doc in docs]
    return list_generate_qa_pairs(list_doc, num_questions_per_page, model=model)


def docs_generate_pdf_qa_pairs(
        pdf_pages: List[Document],
        num_questions_per_page: int = 1,
        model: str = 'glm-4-plus',
) -> QaPairs:
    '''
    借助大模型从给定的texts里提取出问题、答案
    返回结果为问题、答案、所属页码
    '''

    if model not in llm_list:
        raise ValueError('你选择的模型暂时不被支持'
                            '''请使用'glm-4', 'glm-4v', 'glm-3-turbo'中的一个作为model的参数''')
    elif model in llm_list[:3]:
        llm = ZhipuAI(
            api_key="652a160546149ef4e3ec0ff881beebfe.D3UaKuk7FmiUn9WQ"
        )

    qa_pairs = []

    for page in tqdm(pdf_pages):
        if len(page.page_content) > 200:
            # print("line130")
            prompt = PROMPT.format(
                context_str=page.page_content,
                num_questions_per_page=num_questions_per_page
            )
            # print("135")
            try:
                response = llm.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                )
                matches = re.findall(
                    r'问题\d+：(.*?)原文内容\d+：(.*?)((?=问题\d+：)|$)',
                    response.choices[0].message.content,
                    re.DOTALL
                )
                for _, match in enumerate(matches):
                    qa = {
                        'query': match[0].strip(),
                        'answer': match[1].strip(),
                        'page_num': page.metadata['page']
                    }
                    print("问题："+match[0].strip()+"\n")
                    print("答案："+match[1].strip()+"\n")
                    qa_pairs.append(qa)
            except Exception as e:
                print("error")
                continue
    return QaPairs(qa_pairs=qa_pairs)

# 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
loader = PyMuPDFLoader("./knowledge_db/核心考案.pdf")

# 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
pdf_pages = loader.load()
train_pages = pdf_pages[102:329]
qa_pairs = docs_generate_pdf_qa_pairs(train_pages)
qa_pairs.save_json("train_dataset4.json")
#
# qa_from_pdf.save_json("train_dataset.json")
#
# qa_pairs = QaPairs.from_json('./train_dataset.json')
# # 将单个qa对转为InputExample并存入列表
# examples = [InputExample(texts=[qa_pair['query'], qa_pair['answer']]) for qa_pair in qa_pairs.qa_pairs]
# train_examples = examples[:50]
# dev_examples = examples[50:100]
# # 将数据集转换为DataLoader形式
# train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
# print("创建模型")
# model = SentenceTransformer(model_name_or_path="Amu/tao-8k", device="mps", cache_folder='./', trust_remote_code=True)
#
# train_loss = losses.MultipleNegativesRankingLoss(model=model)
# # 实例化评估器，将每次训练后的模型在验证集上测试性能
# print("创建评估器")
# evaluator = BinaryClassificationEvaluator.from_input_examples(dev_examples, name='dev')
# # 定义模型保存路径
# model_save_path='./trained_tao'
# # 微调模型
# print("开始微调")
# model.fit([(train_dataloader, train_loss)],
#           evaluator=evaluator,
#           epochs=1,
#           output_path=model_save_path,
#           )

qa_pairs = QaPairs.from_json('./train_dataset1.json')
template1 = '{"messages": [{"role": "system", "content": "你是一个乐于助人且知识渊博的AI助手。"},{"role": "user", "content": "'
template2 = '"},{"role": "assistant", "content": "'
template3 = '"}]}'
# 将数据转换为JSONL格式并写入文件
with open('data.jsonl', 'w') as jsonl_file:
    for qa_pair in qa_pairs.qa_pairs:
        # 将字典转换为JSON格式的字符串
        jsonl_string = template1+qa_pair['query'].replace('\n', '').replace('"', '').replace('/', '').replace('\\', '')+template2+qa_pair['answer'].replace('\n', '').replace('"', '').replace('/', '').replace('\\', '')+template3
        jsonl_file.write(jsonl_string + '\n')

