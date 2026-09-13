import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


# 1. 配置大模型 (这里以智谱GLM为例，有免费额度)
llm = ChatOpenAI(
    model="glm-4.7",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    openai_api_key="5aa36f72c175460497450b02ca7f586d.4i0riRpLSYlK17d8",
    temperature=0.1
)

# 2. 加载并切分文档
print("正在加载和切分知识库...")
loader = TextLoader("knowledge.txt", encoding="utf-8")
docs = loader.load()

# 使用递归字符切分器，适合中文
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 每个文本块最大长度
    chunk_overlap=50,  # 块与块之间的重叠，避免语义断裂
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
splits = text_splitter.split_documents(docs)
print(f"文档已切分为 {len(splits)} 个片段。")

# 3. 向量化并存入 FAISS 向量数据库
print("正在加载 Embedding 模型并构建向量库...")
# 使用本地中文向量模型，首次运行会自动下载 (约100MB)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 构建向量数据库
vectorstore = FAISS.from_documents(splits, embeddings)
print("向量数据库构建完成！")

# 4. 构建检索器和 RAG 链
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # 检索最相似的2个片段

# 定义 Prompt 模板
system_prompt = (
    "你是一个专业的AI行业知识库助手。"
    "请根据以下提供的上下文来回答用户的问题。"
    "如果你不知道答案，请如实说你不知道，不要编造。"
    "\n\n"
    "上下文：\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 创建 RAG 链
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 5. 测试问答
print("\n--- RAG 问答机器人已启动 (输入 'q' 退出) ---")
while True:
    query = input("\n请输入你的问题: ")
    if query.lower() == 'q':
        break
    if not query.strip():
        continue

    response = rag_chain.invoke({"input": query})
    print("\nAI 回答: ", response["answer"])