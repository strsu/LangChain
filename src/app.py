import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
import tempfile
import os

st.set_page_config(page_title="PDF QA 챗봇", layout="wide")

st.title("📄 PDF 업로드 + 질문 답변 챗봇 (한글 지원)")

# 1. PDF 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_pdf_path = tmp_file.name

    st.success(f"'{uploaded_file.name}' 업로드 완료!")

    # 2. 문서 로드 및 분할
    loader = PyPDFLoader(tmp_pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # 3. 임베딩 & 벡터스토어 초기화
    with st.spinner("임베딩 중입니다. 잠시만 기다려주세요..."):
        embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/clip-ViT-B-32-multilingual-v1")
        vectordb = Chroma.from_documents(documents=docs, embedding=embedding_model, persist_directory="chroma_db")

    # 4. LLM 초기화
    llm = Ollama(model="mistral")

    # 5. Retrieval QA 체인 생성
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever(), return_source_documents=True)

    # 질문 입력 UI
    query = st.text_input("질문을 입력하세요", placeholder="예: 이 문서의 주요 내용은 무엇인가요?")

    if query:
        with st.spinner("답변 생성 중..."):
            result = qa({"query": query})

        st.markdown("### 💬 답변:")
        st.write(result["result"])

        # 출처 문서 보기
        with st.expander("🔍 답변 근거 문서 보기"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**문서 chunk {i+1}:**")
                st.write(doc.page_content)

    # 임시 파일 삭제
    os.remove(tmp_pdf_path)

else:
    st.info("PDF 파일을 업로드해주세요.")
