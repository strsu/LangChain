# LangChain 프로젝트

이 프로젝트는 LangChain을 사용한 문서 처리 및 AI 작업을 위한 환경을 제공합니다.

## 시스템 요구사항

- Python 3.8 이상
- UV 패키지 매니저
- Ollama (시스템 애플리케이션)

## 설치 방법

1. Ollama 설치 (시스템에 없는 경우)
   - [Ollama 공식 웹사이트](https://ollama.ai)에서 다운로드
   - Mac의 경우:
   ```bash
   curl https://ollama.ai/install.sh | sh
   ```
   - Linux의 경우:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
   - Linux의 경우 [Linux 설치 가이드](https://github.com/ollama/ollama/blob/main/docs/linux.md) 참고
   - Windows의 경우 [Windows 설치 가이드](https://github.com/ollama/ollama/blob/main/docs/windows.md) 참고

2. 저장소 클론
```bash
git clone <repository-url>
cd langchain-project
```

3. UV로 가상환경 생성 및 의존성 설치
```bash
uv venv
source .venv/bin/activate  # Linux/Mac
# Windows의 경우: .venv\Scripts\activate

uv pip sync pyproject.toml
```

4. Ollama 모델 다운로드
```bash
ollama pull mistral
```

## 실행 방법

1. Ollama 서비스 실행 확인
```bash
# 새 터미널에서
ollama serve
```

2. 가상환경이 활성화되어 있는지 확인
```bash
# 가상환경 활성화가 되어 있지 않다면:
source .venv/bin/activate  # Linux/Mac
# Windows의 경우: .venv\Scripts\activate
```

3. Streamlit 앱 실행:
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
- `ollama`: Ollama API 클라이언트 (Python 패키지)

## 프로젝트 구조

```
langchain-project/
├── pyproject.toml    # 프로젝트 설정 및 의존성
├── app.py           # Streamlit 애플리케이션 메인 파일
├── .venv/           # 가상환경 디렉토리
└── src/             # 소스 코드
```

## 문제 해결

1. Ollama 관련 문제:
   - Ollama 서비스가 실행 중인지 확인: `ollama serve`
   - Ollama 서비스 상태 확인: `ollama list`
   - 모델이 제대로 설치되었는지 확인: `ollama list`

2. 'streamlit' 명령어를 찾을 수 없는 경우:
   - 가상환경이 활성화되어 있는지 확인
   - PATH에 Python 스크립트 경로가 추가되어 있는지 확인
   - 필요한 경우 `python -m streamlit run app.py` 사용

## 의존성 관리

새로운 패키지를 추가하려면:
1. pyproject.toml 파일에 패키지 추가
2. 의존성 동기화:
```bash
uv pip sync pyproject.toml
```
