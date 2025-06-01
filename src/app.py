import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
import tempfile
import os
import hashlib
import json
from datetime import datetime
from typing import List, Dict
import shutil

# 상수 정의
UPLOAD_DIR = "uploaded_pdfs"
PDF_INFO_FILE = "pdf_info.json"
CHROMA_DIR = "chroma_dbs"

# 디렉토리 생성
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# PDF 정보 로드
def load_pdf_info():
    if os.path.exists(PDF_INFO_FILE):
        with open(PDF_INFO_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# PDF 정보 저장
def save_pdf_info(pdf_info):
    with open(PDF_INFO_FILE, 'w', encoding='utf-8') as f:
        json.dump(pdf_info, f, ensure_ascii=False, indent=2)

# PDF 파일의 해시값 생성
def get_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()

# 선택된 PDF들의 벡터 스토어 결합
def combine_vectorstores(pdf_hashes: List[str], pdf_info: Dict, embedding_model) -> Chroma:
    if len(pdf_hashes) == 1:
        return Chroma(
            persist_directory=pdf_info[pdf_hashes[0]]["chroma_dir"],
            embedding_function=embedding_model
        )
    
    # 임시 디렉토리에 결합된 벡터 스토어 생성
    temp_dir = os.path.join(CHROMA_DIR, "temp_combined")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # 첫 번째 벡터 스토어 복사
    shutil.copytree(pdf_info[pdf_hashes[0]]["chroma_dir"], temp_dir)
    combined_db = Chroma(
        persist_directory=temp_dir,
        embedding_function=embedding_model
    )
    
    # 나머지 벡터 스토어의 데이터 추가
    for pdf_hash in pdf_hashes[1:]:
        db = Chroma(
            persist_directory=pdf_info[pdf_hash]["chroma_dir"],
            embedding_function=embedding_model
        )
        combined_db._collection.add(
            embeddings=db._collection.get()["embeddings"],
            documents=db._collection.get()["documents"],
            metadatas=db._collection.get()["metadatas"],
            ids=db._collection.get()["ids"]
        )
    
    return combined_db

# Streamlit 페이지 설정
st.set_page_config(page_title="PDF QA 챗봇", layout="wide")
st.title("📄 PDF 업로드 + 질문 답변 챗봇 (한글 지원)")

# PDF 정보 로드
pdf_info = load_pdf_info()

# 사이드바에 PDF 목록 표시
st.sidebar.title("📚 저장된 PDF 목록")

# PDF 그룹화 (파일명 기준)
pdf_groups = {}
for pdf_hash, info in pdf_info.items():
    base_name = info["filename"].rsplit(".", 1)[0]  # 확장자 제외
    if base_name not in pdf_groups:
        pdf_groups[base_name] = []
    pdf_groups[base_name].append(pdf_hash)

# 새 PDF 업로드 처리
uploaded_file = st.sidebar.file_uploader("PDF 파일 업로드", type=["pdf"])
if uploaded_file:
    file_content = uploaded_file.read()
    file_hash = get_file_hash(file_content)
    
    if file_hash in pdf_info:
        st.sidebar.warning(f"'{uploaded_file.name}'는 이미 업로드된 파일입니다!")
    else:
        pdf_path = os.path.join(UPLOAD_DIR, f"{file_hash}.pdf")
        with open(pdf_path, 'wb') as f:
            f.write(file_content)
        
        pdf_info[file_hash] = {
            "filename": uploaded_file.name,
            "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "path": pdf_path,
            "chroma_dir": os.path.join(CHROMA_DIR, file_hash)
        }
        save_pdf_info(pdf_info)
        
        with st.spinner("PDF 처리 중..."):
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = splitter.split_documents(documents)
            
            embedding_model = SentenceTransformerEmbeddings(
                model_name="sentence-transformers/clip-ViT-B-32-multilingual-v1"
            )
            
            Chroma.from_documents(
                documents=docs,
                embedding=embedding_model,
                persist_directory=pdf_info[file_hash]["chroma_dir"]
            )
        
        st.sidebar.success(f"'{uploaded_file.name}' 업로드 및 처리 완료!")
        st.rerun()

# PDF 선택 UI
st.sidebar.markdown("---")
st.sidebar.subheader("📑 분석할 PDF 선택")

selected_pdfs = []
for group_name, pdf_hashes in pdf_groups.items():
    st.sidebar.markdown(f"**{group_name}**")
    for pdf_hash in pdf_hashes:
        info = pdf_info[pdf_hash]
        if st.sidebar.checkbox(
            f"{info['filename']} ({info['upload_date']})",
            key=f"select_{pdf_hash}"
        ):
            selected_pdfs.append(pdf_hash)

# 선택된 PDF가 있을 때만 처리
if selected_pdfs:
    # 선택된 PDF 정보 표시
    st.markdown("### 📌 선택된 PDF 목록:")
    for pdf_hash in selected_pdfs:
        st.info(f"- {pdf_info[pdf_hash]['filename']} (업로드: {pdf_info[pdf_hash]['upload_date']})")
    
    # 벡터 스토어 결합
    embedding_model = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/clip-ViT-B-32-multilingual-v1"
    )
    
    vectordb = combine_vectorstores(selected_pdfs, pdf_info, embedding_model)
    
    # LLM 초기화
    llm = Ollama(model="mistral")
    
    # Retrieval QA 체인 생성
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )
    
    # 질문 입력 UI
    query = st.text_input("질문을 입력하세요", placeholder="예: 이 문서들의 주요 내용은 무엇인가요?")
    
    if query:
        with st.spinner("답변 생성 중..."):
            result = qa({"query": query})
        
        st.markdown("### 💬 답변:")
        st.write(result["result"])
        
        # 출처 문서 보기
        with st.expander("🔍 답변 근거 문서 보기"):
            for i, doc in enumerate(result["source_documents"]):
                source_pdf = doc.metadata.get("source", "").split("/")[-1].split(".")[0]
                st.markdown(f"**문서 chunk {i+1} (출처: {pdf_info.get(source_pdf, {}).get('filename', '알 수 없음')}):**")
                st.write(doc.page_content)
    
    # PDF 삭제 기능
    st.sidebar.markdown("---")
    st.sidebar.subheader("🗑 PDF 삭제")
    for pdf_hash in selected_pdfs:
        if st.sidebar.button(f"삭제: {pdf_info[pdf_hash]['filename']}", key=f"delete_{pdf_hash}"):
            os.remove(pdf_info[pdf_hash]["path"])
            shutil.rmtree(pdf_info[pdf_hash]["chroma_dir"], ignore_errors=True)
            del pdf_info[pdf_hash]
            save_pdf_info(pdf_info)
            st.sidebar.success("PDF가 삭제되었습니다.")
            st.rerun()

else:
    st.info("분석할 PDF를 선택하거나 새로운 PDF를 업로드해주세요.")
