# LangChain 프로젝트

이 프로젝트는 LangChain을 사용한 문서 처리 및 AI 작업을 위한 환경을 제공합니다.

## 시스템 요구사항

- Python 3.12 이상
- UV 패키지 매니저
- Ollama

## 설치 방법

1. 저장소 클론
```bash
git clone <repository-url>
cd langchain-project
```

2. 필수 라이브러리 설치 (UV 사용)
```bash
uv pip install streamlit langchain chromadb pypdf sentence-transformers ollama
```
또는 pip 사용:
```bash
pip install streamlit langchain chromadb pypdf sentence-transformers ollama
```

3. Ollama 모델 설치
```bash
ollama pull mistral
```

## 실행 방법

1. Streamlit 앱 실행:
```bash
streamlit run app.py
```
실행 후 자동으로 브라우저가 열리며 기본 주소는 `http://localhost:8501` 입니다.

## 주요 의존성 패키지

- `streamlit`: 웹 인터페이스 구현
- `langchain`: LLM 애플리케이션 개발을 위한 프레임워크
- `chromadb`: 벡터 데이터베이스
- `pypdf`: PDF 파일 처리
- `sentence-transformers`: 텍스트 임베딩
- `ollama`: 로컬 LLM 실행

## 프로젝트 구조

```
langchain-project/
├── pyproject.toml    # 프로젝트 설정 및 의존성
├── app.py           # Streamlit 애플리케이션 메인 파일
└── src/             # 소스 코드
```

## 문제 해결

1. Ollama 설치가 필요한 경우:
   - [Ollama 공식 웹사이트](https://ollama.ai)에서 설치
   - Mac: `curl https://ollama.ai/install.sh | sh`

2. 'streamlit' 명령어를 찾을 수 없는 경우:
   - PATH에 Python 스크립트 경로가 추가되어 있는지 확인
   - 필요한 경우 `python -m streamlit run app.py` 사용

## 의존성 관리

새로운 패키지를 추가하려면:
```bash
uv pip install <package-name>
```

의존성을 업데이트하려면:
```bash
uv pip install --upgrade <package-name>
```
