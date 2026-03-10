# PDF-to-EPUB Converter — 이슈 정리

최종 업데이트: 2026-03-10

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
- PaddleX (PP-DocLayout-M)가 게이트키퍼 역할
- 페이지 내 영역을 text, figure, table, header, footer 등으로 분류
- 분류 결과에 따라 각 영역별 처리 방식 결정

### 4.3 문단 복원 전략
- **Born-digital (pdf_extractor.py)**: bbox 기반 — per-line bbox + 오른쪽 마진 + short-line 감지
- **Scanned (structure_parser.py)**: text-level 휴리스틱 — 줄 길이 비율 + 구두점 + 들여쓰기

---

## 5. 미해결 과제

### 5.1 PP-DocLayout-M 카테고리 인식 문제 (부분 해결)
- **작동 확인**: text, number, header, image (figure)
- **해결**: 모델이 figure를 `"image"` 레이블로 출력 — `FIGURE_TYPES`에 `"image"` 추가하여 해결
- **미확인**: table, footer, 기타 (23개 중 대부분 미검증)
- **다음 단계**: 다양한 레이아웃의 테스트 PDF로 나머지 카테고리 체계적 테스트

### 5.2 삼중 OCR 투표 시스템
- 아키텍처 결정은 완료, 구현 및 실제 검증 필요

### 5.3 EPUB 출력 품질
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
