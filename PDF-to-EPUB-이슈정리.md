# PDF-to-EPUB Converter — 이슈 정리

최종 업데이트: 2026-03-12

---

## 1. 환경 및 의존성 이슈

### 1.1 requirements.txt CP949 인코딩 (해결)
- **증상**: em dash (—) 문자가 CP949 인코딩에서 깨짐
- **해결**: em dash를 일반 hyphen (-)으로 교체

### 1.2 numpy 버전 충돌 (해결)
- **증상**: PaddlePaddle 설치 시 numpy 2.2.6으로 업그레이드, 기존 프로젝트와 충돌
- **해결**: Anaconda 가상환경 `pdfepub` 생성하여 격리
- **활성화**: `conda activate pdfepub`

### 1.3 PaddlePaddle oneDNN 비호환 (해결 — 중요)
- **증상**: PaddlePaddle 3.3.0에서 oneDNN 관련 크래시 발생
- **해결**: PaddlePaddle 3.0.0으로 다운그레이드
- **주의**: 절대 PaddlePaddle을 3.0.0 이상으로 업그레이드하지 말 것

### 1.4 Tesseract PATH (해결)
- **증상**: Python에서 Tesseract 실행 파일을 찾지 못함
- **해결**: Windows 시스템 환경 변수에서 GUI를 통해 수동으로 PATH 추가

---

## 2. API 및 라이브러리 변경 이슈

### 2.1 PaddleOCR API 변경 (해결)
- `show_log` 파라미터 제거됨 → 코드에서 삭제
- `.ocr()` 메서드 → `.predict()`로 변경

### 2.2 PyMuPDF API (해결)
- `get_text("rawdict")` → `get_text("dict")`로 변경
- `get_clip_text()` → `get_paragraph_text()`로 교체 (bbox 기반 문단 복원)

### 2.3 PaddleX 파이프라인 및 모델 (해결)
- Pipeline name: `layout_detection` → `layout_parsing`
- 모델: `PicoDet_layout_1x` 실패 → **PP-DocLayout-M** 사용
  - 75.2% mAP
  - 23개 카테고리 지원
  - PaddlePaddle 3.0.0 CPU 호환 확인
- 환경변수: PaddleX import 전에 `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` 설정 필요

### 2.4 PaddleX v3 결과 파싱 구조 (해결)
- `_parse_paddlex_result()`가 읽는 구조:
  ```
  item['layout_det_res']['boxes'][i]
  ├── coordinate  (bbox 좌표)
  ├── label       (카테고리 라벨)
  └── score       (신뢰도 점수)
  ```
- 하위 호환: 이전 flat 형식 `boxes/labels/scores`도 fallback으로 유지

---

## 3. 코드 구조 변경 이력

### 3.1 epub_builder.py
- `uid=uid` (undefined variable) → `uid=f"image-{img.image_id}"`

### 3.2 pdf_extractor.py
- `get_paragraph_text()` 추가:
  - `get_text("dict")`로 per-line bbox 추출
  - 95th-percentile x1 값으로 오른쪽 마진 추정
  - short-line (< 82% of right margin) + 문장 끝 구두점으로 문단 병합

### 3.3 main.py
- `get_clip_text()` → `get_paragraph_text()`로 교체
- born-digital 페이지에서 bbox 기반 문단 복원 적용
- `--pages` CLI 옵션 추가: 특정 페이지 또는 범위만 처리 (예: `--pages 50`, `--pages 10-20`)

### 3.4 structure_parser.py
- `_restore_paragraphs()` 추가:
  - 스캔 페이지용 text-level 휴리스틱
  - short-line (< 70% of median) + 문장 끝 구두점 + 들여쓰기 감지
  - 각주 마커 없는 plain text region에 적용
- `build_blocks()`: `_restore_paragraphs()` 적용하여 OCR 라인 단위 대신 복원된 문단 단위로 `<p>` 블록 생성

---

## 4. 아키텍처 결정 사항

### 4.1 삼중 OCR 투표 시스템
- **A**: PyMuPDF (네이티브 텍스트 추출)
- **B**: Tesseract OCR
- **C**: EasyOCR (`["ko", "en"]`)
- **비교 방식**: `difflib.SequenceMatcher`로 문자 단위 정렬
- **정렬 구조**: A가 앵커 → A-B 정렬, A-C 정렬 (양자 정렬)
- **투표 규칙**:
  - 3개 후보 모두 존재 → 다수결
  - 후보 부재 시 → 네이티브 텍스트로 fallback하지 않고 에러 로그 기록
- **설계 원칙**: 네이티브 텍스트(A)가 더 정확하다는 가정 없음. 투표 시스템이 A 단독보다 나은 결과를 내야 함.

### 4.2 레이아웃 분석 역할
- **Surya** (1순위) 또는 PaddleX (fallback)가 게이트키퍼 역할
- 페이지 내 영역을 text, figure, table, header, footer 등으로 분류
- 분류 결과에 따라 각 영역별 처리 방식 결정
- PP-DocLayout-M → Surya 교체 이유: PP-DocLayout-M이 figure와 text를 반대로 분류하는 치명적 오류 발견

### 4.3 문단 복원 전략
- **Born-digital (pdf_extractor.py)**: bbox 기반 — per-line bbox + 오른쪽 마진 + short-line 감지
- **Scanned (structure_parser.py)**: text-level 휴리스틱 — 줄 길이 비율 + 구두점 + 들여쓰기

---

## 5. 해결된 이슈 (2026-03-11)

### 5.1 PP-DocLayout-M → Surya 교체 (해결)
- **증상**: PP-DocLayout-M이 figure(그림)를 text로, text를 image로 반대 분류
- **디버깅**: `debug_layout.py`로 bbox 시각화하여 확인
- **해결**: Surya LayoutPredictor로 교체 (`surya-ocr` 0.16.0)
  - 14개 카테고리: Text, Picture, Figure, Caption, SectionHeader, PageHeader, PageFooter 등
  - 라벨 정규화: PascalCase → lowercase (예: `Picture` → `picture`)
  - PaddleX는 fallback으로 유지
- **주의**: `surya-ocr` 0.17.1은 `transformers` 5.x와 호환 문제 발생 → 0.16.0 사용

### 5.2 EPUB 이미지 렌더링 (해결)
- **증상 1**: 이미지가 `img_0001` 텍스트로 표시됨
  - **원인**: `structure_parser.py`의 figure type 체크에 `"image"`, `"picture"` 누락
  - **해결**: `FIGURE_REGION_TYPES` 공유 set 도입하여 불일치 방지
- **증상 2**: 이미지 태그는 생성되나 로딩 실패
  - **원인**: `epub_builder.py`에서 `src="../images/"` 경로 오류 (content.xhtml이 루트에 있으므로 `../` 불필요)
  - **해결**: `src="images/"` 로 수정

### 5.3 페이지 하단 제목/번호가 본문에 삽입되는 문제 (부분 해결)
- **해결 1**: Surya가 PageHeader/PageFooter를 정확히 분류 → DISCARD_TYPES에 추가
- **해결 2**: 위치 기반 휴리스틱 추가 — 페이지 상단 8% / 하단 10% 에 있는 짧은 텍스트(높이 < 3%) 자동 discard
- **미해결**: 여러 페이지 간 반복 텍스트 패턴 감지 (Phase 2)

### 5.4 GitHub SSH 인증 설정 (해결)
- Linux 환경에서 HTTPS 인증 실패 → SSH 키 생성 및 등록
- remote URL: `https://` → `git@github.com:` 변경

---

## 6. 미해결 과제

### 6.1 텍스트 품질 향상 — OCR 투표 시스템 (최우선)
- 현재 OCR 출력 품질이 낮음: 글자 오인식, 띄어쓰기 손실, 문장 누락
- **삼중 OCR 투표 시스템**: 아키텍처 결정은 완료, 구현 및 검증 필요
- **Surya OCR 추가 검토**: 4번째 후보로 추가하여 4중 투표 시스템으로 확장 가능
- **GLM-OCR 조사**: Reddit에서 좋은 평가, AI 기반 OCR로 추가 후보 가능성

### 6.2 여러 페이지 간 반복 텍스트 패턴 감지
- header/footer가 모델에 의해 text로 분류될 경우, 여러 페이지에서 동일 텍스트가 반복되면 discard
- `main.py` 파이프라인 레벨에서 구현 필요

### 6.3 Linux 환경에서 Surya/PaddleX 호환성
- Linux에서 Surya 동작 여부 미검증
- PaddleX는 Linux에서 초기화 실패 이력 있음 (paddlepaddle 3.3.0)

### 6.4 EPUB 출력 품질
- reflowable EPUB의 실제 렌더링 품질 검증 미완료

---

## 6. 참고: EasyOCR 언어 설정

```python
# 올바른 설정
reader = easyocr.Reader(["ko", "en"])

# 잘못된 설정 (ch_tra 포함하면 오류)
reader = easyocr.Reader(["ko", "en", "ch_tra"])  # ← 사용 금지
```

---

## 7. 참고: 환경변수 설정

```python
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
# 이 설정은 반드시 PaddleX import 전에 실행
```
