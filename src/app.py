import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import tempfile
import os
import hashlib
import json
from datetime import datetime
from typing import List, Dict
import shutil
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
import pandas as pd
import nltk
from collections import defaultdict
import requests
import subprocess
import time

# NLTK 데이터 다운로드
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 상수 정의
UPLOAD_DIR = "uploaded_pdfs"
PDF_INFO_FILE = "pdf_info.json"
CHROMA_DIR = "chroma_dbs"
CHAT_HISTORY_FILE = "chat_history.json"
GENERAL_CHAT_KEY = "general_chat"  # 일반 대화용 키

# 사용 가능한 모델 목록
AVAILABLE_MODELS = {
    "tinyllama": "가벼운 모델 (512MB)",
    "llama2": "중간 크기 모델 (3GB)",
    "mistral": "큰 모델 (4GB)",
    "neural-chat": "작은 대화 특화 모델 (1.5GB)"
}

# Ollama 모델 설치 확인 및 설치
def check_and_install_model(model_name: str) -> bool:
    # Ollama 서비스 확인
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            return False
    except requests.exceptions.ConnectionError:
        st.error("""
        Ollama 서비스가 실행되고 있지 않습니다. 다음 단계를 따라주세요:
        
        1. Ollama 설치 (처음 실행 시):
        ```bash
        curl -fsSL https://ollama.com/install.sh | sh
        ```
        
        2. Ollama 서비스 실행:
        ```bash
        ollama serve
        ```
        """)
        return False
    
    # 설치된 모델 확인
    installed_models = [tag["name"] for tag in response.json().get("models", [])]
    if model_name not in installed_models:
        with st.spinner(f"'{model_name}' 모델 설치 중... (처음 실행시 몇 분 소요될 수 있습니다)"):
            try:
                subprocess.run(["ollama", "pull", model_name], check=True)
                time.sleep(2)  # 설치 완료 후 잠시 대기
                return True
            except subprocess.CalledProcessError:
                st.error(f"""
                '{model_name}' 모델 설치 중 오류가 발생했습니다.
                터미널에서 다음 명령어를 실행해주세요:
                ```bash
                ollama pull {model_name}
                ```
                """)
                return False
    return True

def is_model_installed(model_name: str) -> bool:
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            installed_models = [tag["name"] for tag in response.json().get("models", [])]
            return model_name in installed_models
    except requests.exceptions.ConnectionError:
        return False
    return False

# 디렉토리 생성
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# 채팅 히스토리 관리 함수
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return defaultdict(list)

def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# 세션 상태 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = load_chat_history()

# Streamlit 페이지 설정
st.set_page_config(page_title="AI 챗봇", layout="wide")

# 사이드바에 모델 선택 추가
st.sidebar.title("🤖 모델 설정")
selected_model = st.sidebar.selectbox(
    "사용할 모델 선택",
    options=list(AVAILABLE_MODELS.keys()),
    format_func=lambda x: f"{x} - {AVAILABLE_MODELS[x]}",
    index=0  # 기본값으로 tinyllama 선택
)

# 모델 설치 확인
if not check_and_install_model(selected_model):
    st.stop()

# LLM 초기화
@st.cache_resource
def get_llm(model_name):
    return Ollama(model=model_name)

llm = get_llm(selected_model)

# 모델 상태에 따른 사이드바 정보 표시
sidebar_info = f"""
현재 환경: CPU 2코어, RAM 16GB
선택된 모델: {selected_model}
모델 설명: {AVAILABLE_MODELS[selected_model]}
"""

if not is_model_installed(selected_model):
    sidebar_info += f"""
💡 모델 관리 명령어:
```bash
# 모델 설치
ollama pull {selected_model}

# 설치된 모델 목록 확인
ollama list

# 모델 제거
ollama rm {selected_model}
```
"""

st.sidebar.info(sidebar_info)

# 일반 대화용 프롬프트 템플릿
general_chat_prompt = PromptTemplate(
    template="""당신은 한국어로 대화하는 AI 어시스턴트입니다.
다음 사항을 지켜주세요:
1. 항상 한국어로 자연스럽게 대화하기
2. 번역하지 않고 바로 한국어로 생각하고 답변하기
3. 친절하고 전문적으로 답변하기
4. 필요한 경우 예시나 구체적인 설명 추가하기

사용자의 질문: {question}

답변:""",
    input_variables=["question"]
)

general_chat_chain = LLMChain(
    llm=llm,
    prompt=general_chat_prompt
)

# 문서 분석 함수들
def extract_keywords(text: str, top_n: int = 10) -> List[tuple]:
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, 
                                       keyphrase_ngram_range=(1, 2),
                                       stop_words='english', 
                                       top_n=top_n)
    return keywords

def generate_wordcloud(text: str):
    wordcloud = WordCloud(width=800, height=400,
                         background_color='white',
                         font_path='/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
                         ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

def calculate_document_similarity(docs1, docs2, embedding_model):
    # 문서 임베딩 계산
    embeddings1 = embedding_model.embed_documents([doc.page_content for doc in docs1])
    embeddings2 = embedding_model.embed_documents([doc.page_content for doc in docs2])
    
    # 코사인 유사도 계산
    similarity = cosine_similarity(embeddings1, embeddings2)
    return np.mean(similarity)

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

# 탭 생성
tab1, tab2, tab3, tab4 = st.tabs(["💬 일반 대화", "📄 PDF 분석", "📊 문서 분석", "📝 대화 기록"])

# 일반 대화 탭
with tab1:
    st.title("💬 AI 챗봇과 대화하기")
    
    chat_input = st.text_input("무엇이든 물어보세요", key="general_chat_input")
    
    if chat_input:
        with st.spinner("답변 생성 중..."):
            response = general_chat_chain.invoke({"question": chat_input})
            
            st.markdown("### 💬 답변:")
            st.write(response['text'])  # invoke는 딕셔너리를 반환하므로 'text' 키로 접근
            
            # 대화 히스토리 저장
            st.session_state.chat_history[GENERAL_CHAT_KEY].append({
                'question': chat_input,
                'answer': response['text'],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            save_chat_history(st.session_state.chat_history)

# PDF 분석 탭
with tab2:
    st.title("📄 PDF 업로드 + 질문 답변")
    
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
                
                embedding_model = HuggingFaceEmbeddings(
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
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/clip-ViT-B-32-multilingual-v1"
        )
        
        vectordb = combine_vectorstores(selected_pdfs, pdf_info, embedding_model)
        
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
                result = qa.invoke({"query": query})
            
            st.markdown("### 💬 답변:")
            st.write(result["result"])
            
            # 출처 문서 보기
            with st.expander("🔍 답변 근거 문서 보기"):
                for i, doc in enumerate(result["source_documents"]):
                    source_pdf = doc.metadata.get("source", "").split("/")[-1].split(".")[0]
                    st.markdown(f"**문서 chunk {i+1} (출처: {pdf_info.get(source_pdf, {}).get('filename', '알 수 없음')}):**")
                    st.write(doc.page_content)
            
            # 채팅 히스토리에 대화 추가
            chat_key = ','.join(sorted(selected_pdfs))
            st.session_state.chat_history[chat_key].append({
                'question': query,
                'answer': result["result"],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            save_chat_history(st.session_state.chat_history)

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

# 문서 분석 탭
with tab3:
    st.title("📊 문서 분석")
    
    if selected_pdfs:
        # 분석 옵션 선택
        analysis_type = st.selectbox(
            "분석 유형 선택",
            ["문서 요약", "키워드 분석", "문서 유사도 분석", "워드클라우드"]
        )
        
        if analysis_type == "문서 요약":
            with st.spinner("문서 요약 중..."):
                summary_prompt = PromptTemplate(
                    template="다음 문서를 500자 이내로 요약해주세요:\n\n{text}\n\n요약:",
                    input_variables=["text"]
                )
                
                summary_chain = LLMChain(
                    llm=llm,
                    prompt=summary_prompt
                )
                
                for pdf_hash in selected_pdfs:
                    st.subheader(f"📑 {pdf_info[pdf_hash]['filename']}")
                    loader = PyPDFLoader(pdf_info[pdf_hash]['path'])
                    documents = loader.load()
                    text = "\n".join([doc.page_content for doc in documents])
                    summary_result = summary_chain.invoke({"text": text})
                    st.write(summary_result['text'])
        
        elif analysis_type == "키워드 분석":
            with st.spinner("키워드 추출 중..."):
                for pdf_hash in selected_pdfs:
                    st.subheader(f"📑 {pdf_info[pdf_hash]['filename']}")
                    loader = PyPDFLoader(pdf_info[pdf_hash]['path'])
                    documents = loader.load()
                    text = "\n".join([doc.page_content for doc in documents])
                    keywords = extract_keywords(text)
                    
                    # 키워드 시각화
                    df = pd.DataFrame(keywords, columns=['Keyword', 'Score'])
                    st.bar_chart(df.set_index('Keyword')['Score'])
        
        elif analysis_type == "문서 유사도 분석":
            if len(selected_pdfs) < 2:
                st.warning("문서 유사도 분석을 위해서는 2개 이상의 PDF를 선택해주세요.")
            else:
                with st.spinner("문서 유사도 분석 중..."):
                    similarity_matrix = []
                    for i, pdf1 in enumerate(selected_pdfs):
                        row = []
                        for j, pdf2 in enumerate(selected_pdfs):
                            if i == j:
                                row.append(1.0)
                            elif j > i:
                                loader1 = PyPDFLoader(pdf_info[pdf1]['path'])
                                loader2 = PyPDFLoader(pdf_info[pdf2]['path'])
                                docs1 = loader1.load()
                                docs2 = loader2.load()
                                similarity = calculate_document_similarity(docs1, docs2, embedding_model)
                                row.append(similarity)
                            else:
                                row.append(similarity_matrix[j][i])
                        similarity_matrix.append(row)
                    
                    # 유사도 행렬 시각화
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(similarity_matrix, cmap='YlOrRd')
                    
                    # 축 레이블 설정
                    filenames = [pdf_info[pdf]['filename'] for pdf in selected_pdfs]
                    ax.set_xticks(np.arange(len(filenames)))
                    ax.set_yticks(np.arange(len(filenames)))
                    ax.set_xticklabels(filenames, rotation=45, ha='right')
                    ax.set_yticklabels(filenames)
                    
                    # 컬러바 추가
                    plt.colorbar(im)
                    plt.tight_layout()
                    st.pyplot(fig)
        
        elif analysis_type == "워드클라우드":
            with st.spinner("워드클라우드 생성 중..."):
                for pdf_hash in selected_pdfs:
                    st.subheader(f"📑 {pdf_info[pdf_hash]['filename']}")
                    loader = PyPDFLoader(pdf_info[pdf_hash]['path'])
                    documents = loader.load()
                    text = "\n".join([doc.page_content for doc in documents])
                    fig = generate_wordcloud(text)
                    st.pyplot(fig)

# 대화 기록 탭
with tab4:
    st.title("📝 대화 기록")
    
    # 대화 유형 선택
    chat_type = st.radio(
        "대화 유형 선택",
        ["일반 대화", "PDF 분석 대화"],
        horizontal=True
    )
    
    if chat_type == "일반 대화":
        if GENERAL_CHAT_KEY in st.session_state.chat_history and st.session_state.chat_history[GENERAL_CHAT_KEY]:
            for chat in st.session_state.chat_history[GENERAL_CHAT_KEY]:
                with st.expander(f"🕒 {chat['timestamp']} - {chat['question'][:50]}..."):
                    st.markdown("**질문:**")
                    st.write(chat['question'])
                    st.markdown("**답변:**")
                    st.write(chat['answer'])
                    
                    # 답변 평가 버튼
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👍 도움됨", key=f"helpful_general_{chat['timestamp']}"):
                            st.success("피드백 감사합니다!")
                    with col2:
                        if st.button("👎 도움안됨", key=f"not_helpful_general_{chat['timestamp']}"):
                            st.error("피드백 감사합니다!")
        else:
            st.info("아직 일반 대화 기록이 없습니다.")
    
    else:  # PDF 분석 대화
        if st.session_state.chat_history:
            for pdf_group, history in st.session_state.chat_history.items():
                if pdf_group == GENERAL_CHAT_KEY:  # 일반 대화는 건너뛰기
                    continue
                    
                if not history:  # 빈 히스토리는 건너뛰기
                    continue
                
                pdf_names = [
                    pdf_info[pdf_hash]['filename'] 
                    for pdf_hash in pdf_group.split(',') 
                    if pdf_hash in pdf_info
                ]
                if not pdf_names:  # PDF가 삭제된 경우 건너뛰기
                    continue
                    
                st.subheader(f"📚 문서: {', '.join(pdf_names)}")
                
                for chat in history:
                    with st.expander(f"🕒 {chat['timestamp']} - {chat['question'][:50]}..."):
                        st.markdown("**질문:**")
                        st.write(chat['question'])
                        st.markdown("**답변:**")
                        st.write(chat['answer'])
                        
                        # 답변 평가 버튼
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("👍 도움됨", key=f"helpful_{chat['timestamp']}"):
                                st.success("피드백 감사합니다!")
                        with col2:
                            if st.button("👎 도움안됨", key=f"not_helpful_{chat['timestamp']}"):
                                st.error("피드백 감사합니다!")
        else:
            st.info("아직 PDF 분석 대화 기록이 없습니다.")
