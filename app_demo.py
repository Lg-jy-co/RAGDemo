import os
import json
import tempfile
import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


# 用户配置
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()

# 核心函数：构建向量库
@st.cache_resource(show_spinner=False)
def build_vectorstore(files_info):
    """根据上传的文件构建 FAISS 向量数据库"""
    documents = []

    for file_name, file_bytes in files_info:
        # 将上传的文件写入临时文件以便加载器读取
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        try:
            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
            else:
                loader = TextLoader(temp_path, encoding="utf-8")

            documents.extend(loader.load())
        finally:
            os.remove(temp_path)  # 清理临时文件

    if not documents:
        return None

    # 文本切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    )
    splits = text_splitter.split_documents(documents)

    # 向量化（使用最新的 langchain-huggingface 包）
    embeddings = HuggingFaceEmbeddings(model_name="./bge-small-zh")

    # 构建 FAISS 向量库
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

# 页面配置
st.set_page_config(page_title="AI 行业知识库问答", page_icon="🤖", layout="wide")
st.title("🤖 基于 RAG 的行业知识库问答系统")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置面板")

    # FreeLLMAPI 配置
    api_base = st.text_input("API Base URL", value=config.get("api_base", "https://open.bigmodel.cn/api/paas/v4/"))
    api_key = st.text_input("API Key", value=config.get("api_key", ""), type="password")
    model_name = st.text_input("模型名称", value=config.get("model", "glm-4.7"))

    st.divider()
    st.header("📁 知识库上传")
    uploaded_files = st.file_uploader("上传知识库文件 (支持 TXT, PDF)", type=["txt", "pdf"], accept_multiple_files=True)

    if st.button("构建/更新知识库", type="primary"):
        if not uploaded_files:
            st.warning("请先上传文件")
        else:
            with st.spinner("正在处理知识库文件，构建向量索引中..."):
                files_info = [(f.name, f.getvalue()) for f in uploaded_files]
                vectorstore = build_vectorstore(tuple(files_info))
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.vectorstore_ready = True
                    st.success("知识库构建完成！")
                else:
                    st.error("文件解析失败，请检查文件内容。")

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

# 显示历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 查看引用来源"):
                for i, doc in enumerate(message["sources"]):
                    st.caption(f"片段 {i + 1}: {doc.page_content[:200]}...")

# 处理用户输入
if prompt := st.chat_input("请输入你的问题..."):

    # 检查知识库是否就绪
    if not st.session_state.get("vectorstore_ready"):
        st.warning("请先在左侧上传文件并构建知识库！")
    else:
        # 1. 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. 构建 RAG 链并生成回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                # 初始化 LLM
                llm = ChatOpenAI(
                    model=config["model"],
                    openai_api_key=config["api_key"],
                    openai_api_base=config["api_base"],
                    temperature=0.1
                )

                # 构建 Prompt，包含上下文和历史对话
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "你是一个专业的AI行业知识库助手。请根据以下提供的上下文来回答用户的问题。如果你不知道答案，请如实说你不知道，不要编造。\n\n上下文：\n{context}"),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ])

                # 构建检索链
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                # 转换历史对话格式
                chat_history = []
                for msg in st.session_state.messages[:-1]:  # 排除刚加入的当前问题
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"]))

                # 执行 RAG 链
                try:
                    response = rag_chain.invoke({
                        "input": prompt,
                        "chat_history": chat_history
                    })
                    answer = response["answer"]
                    sources = response.get("context", [])

                    # 显示答案
                    st.markdown(answer)

                    # 显示引用来源
                    if sources:
                        with st.expander("📚 查看引用来源"):
                            for i, doc in enumerate(sources):
                                st.caption(f"片段 {i + 1}: {doc.page_content[:200]}...")

                    # 保存到 Session State
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"调用大模型失败，请检查 FreeLLMAPI 是否启动。错误信息: {e}")