# LangChain 프로젝트

이 프로젝트는 LangChain을 사용한 문서 처리 및 AI 작업을 위한 환경을 제공합니다.

## 시스템 요구사항

- Python 3.9 이상 (streamlit 1.45.1 요구사항)
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

2. 저장소 클론
```bash
git clone <repository-url>
cd langchain-project
```

3. 가상환경 생성 및 의존성 설치
```bash
# 가상환경 생성
python3.9 -m venv .venv  # Python 3.9 이상 사용

# 가상환경 활성화
. .venv/bin/activate  # Linux/Mac
# Windows의 경우: .venv\Scripts\activate

# 의존성 설치 (두 가지 방법 중 선택)
uv pip install -e .     # 개발 모드 설치 (권장)
# 또는
uv pip sync pyproject.toml  # 정확한 의존성 동기화
```

4. Ollama 모델 다운로드
```bash
ollama pull mistral
```

## 의존성 패키지

주요 패키지:
- `streamlit>=1.45.1`: 웹 인터페이스 구현 (Python 3.9+ 필요)
- `langchain`: LLM 애플리케이션 개발 프레임워크
- `langchain-community`: LangChain 커뮤니티 통합 패키지
- `chromadb`: 벡터 데이터베이스
- `pypdf`: PDF 파일 처리
- `sentence-transformers`: 텍스트 임베딩
- `ollama`: Ollama API 클라이언트

지원 패키지:
- `blinker`: 시그널 처리
- `protobuf`: 데이터 직렬화
- `typing_extensions`: 타입 힌트 확장
- `cachetools>=4.0,<6.0`: 캐시 관리
- `click`: CLI 도구
- `tornado`: 웹 서버
- `requests`: HTTP 클라이언트
- `urllib3`: HTTP 클라이언트 라이브러리
- `idna`: 국제화 도메인 처리
- `certifi`: SSL/TLS 인증서

## 실행 방법

1. Ollama 서비스 실행 확인
```bash
# 새 터미널에서
ollama serve
```

2. 가상환경 활성화 확인
```bash
# 가상환경이 활성화되어 있지 않다면:
. .venv/bin/activate  # Linux/Mac
# Windows의 경우: .venv\Scripts\activate
```

3. Streamlit 앱 실행:
```bash
python -m streamlit run src/app.py
```
실행 후 자동으로 브라우저가 열리며 기본 주소는 `http://localhost:8501` 입니다.

## 프로젝트 구조

```
langchain-project/
├── pyproject.toml    # 프로젝트 설정 및 의존성
├── README.md        # 프로젝트 문서
├── start.sh         # 실행 스크립트
├── app.py           # Streamlit 애플리케이션 메인 파일
├── .venv/           # 가상환경 디렉토리
└── src/             # 소스 코드
```

## 문제 해결

1. Python 버전 문제:
   - streamlit 1.45.1은 Python 3.9 이상을 요구합니다
   - 가상환경 생성 시 `python3.9` 명령어를 사용하세요

2. 의존성 설치 방법 차이:
   - `uv pip install -e .`: 개발 모드로 설치, 필요한 의존성 자동 추가
   - `uv pip sync pyproject.toml`: pyproject.toml에 명시된 의존성만 정확히 설치

3. Ollama 관련 문제:
   - Ollama 서비스가 실행 중인지 확인: `ollama serve`
   - 모델 상태 확인: `ollama list`

4. Streamlit 실행 문제:
   - 가상환경이 활성화되어 있는지 확인
   - `python -m streamlit run src/app.py` 사용
