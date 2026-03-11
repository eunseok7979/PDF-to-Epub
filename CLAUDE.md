# CLAUDE.md — PDF-to-EPUB Converter

## 작업 규칙 (MUST READ FIRST)

### 대화 스타일
- 새로운 지시를 받으면 **코드를 짜기 전에 반드시**:
  1. 내가 뭘 원하는지 자기 말로 요약해서 확인할 것
  2. 불확실한 점이 있으면 질문할 것
  3. 접근 방식을 제안하고 승인받은 후에 코딩 시작
- 사용자가 "그냥 해" 또는 "바로 짜줘"라고 명시적으로 말한 경우에만 확인 없이 코딩할 것

### 변경 범위 제한
- 한 번에 100줄 이상 변경하지 말 것 — 단계별로 나눠서 진행
- 변경 전에 영향받는 파일 목록을 먼저 보여줄 것
- 기존 코드를 삭제하거나 대체할 때는 반드시 이유를 설명할 것

### 언어
- 한국어로 대화할 것 (존댓말)
- 코드 주석은 영어로 작성할 것
- 커밋 메시지는 영어로 작성할 것

---

## 프로젝트 개요

스캔된 PDF(한국어/영어)를 reflowable EPUB으로 변환하는 도구.
삼중 OCR 시스템 + 레이아웃 분석을 결합한 구조.

### 환경
- Windows 11, AMD Ryzen 8600G / Radeon 760M
- Anaconda 가상환경: `conda activate pdfepub`
- Python (가상환경 내)
- PaddlePaddle 3.0.0 (CPU, oneDNN 비호환 이슈로 3.3.0 사용 불가)

### 핵심 아키텍처

```
PDF 입력
  │
  ├─ Surya (1순위) / PaddleX (fallback) ─→ 레이아웃 분석 (게이트키퍼)
  │   Surya: 14개 카테고리 (Picture, Text, Caption, PageHeader 등)
  │   surya-ocr 0.16.0 (0.17.1은 transformers 호환 문제)
  │   위치 기반 header/footer 필터링: 상단 8% / 하단 10%
  │
  ├─ A: PyMuPDF ──→ 네이티브 텍스트 추출 (앵커 역할)
  ├─ B: Tesseract ─→ OCR 텍스트
  └─ C: EasyOCR ──→ OCR 텍스트 (["ko", "en"], "ch_tra" 제외)
        │
        ▼
  텍스트 비교 (difflib.SequenceMatcher, 문자 단위)
  A를 앵커로 A-B, A-C 양자 정렬 → 투표
        │
        ▼
  EPUB 생성
```

### 투표 시스템 규칙
- A(PyMuPDF)가 더 정확하다는 가정 없음 — 투표 시스템이 네이티브 텍스트보다 나은 결과를 내야 함
- 세 후보 모두 있을 때: 다수결 투표
- 후보가 없을 때: 네이티브 텍스트로 fallback하지 않고 에러 로그 기록

### 주요 파일 구조
- `main.py` — 진입점, `get_paragraph_text()` 사용
- `pdf_extractor.py` — PDF에서 텍스트/이미지 추출, `get_paragraph_text()` (bbox 기반 단락 복원)
- `epub_builder.py` — EPUB 파일 생성, `uid=f"image-{img.image_id}"`
- `structure_parser.py` — 레이아웃 분석 결과 파싱, `_restore_paragraphs()`, `build_blocks()`
- `requirements.txt` — 의존성 (UTF-8 인코딩 필수, em dash 주의)

### 참고 문서
- `PDF-to-EPUB-이슈정리.md` — 해결된 이슈와 현재 상태 상세 기록

---

## 해결된 주요 이슈 (요약)

| 이슈 | 해결 |
|------|------|
| requirements.txt CP949 인코딩 | em dash → hyphen, UTF-8 저장 |
| numpy 버전 충돌 | pdfepub 전용 가상환경 생성 |
| PaddleOCR API 변경 | `show_log` 제거, `.ocr()` → `.predict()` |
| PaddlePaddle 3.3.0 oneDNN 비호환 | 3.0.0으로 다운그레이드 |
| PaddleX 파이프라인명 | `layout_detection` → `layout_parsing` |
| 레이아웃 모델 | PP-DocLayout-M이 figure/text 반대 분류 → **Surya로 교체** |
| PyMuPDF API | `get_text("rawdict")` → `get_text("dict")` |
| Tesseract PATH | Windows GUI에서 수동 추가 |
| EasyOCR 언어 설정 | `["ko", "en"]` (ch_tra 제외) |
| EPUB 이미지 | `src` 경로 오류 + figure type 불일치 수정 |
| header/footer 혼입 | Surya 분류 + 위치 기반 휴리스틱(상단8%/하단10%) |

## 현재 상태

- **레이아웃 분석**: Surya (1순위) — figure, text, caption, header/footer 정확 분류 확인
- **EPUB 이미지**: 정상 렌더링 확인 (50페이지 테스트)
- **최우선 과제**: OCR 텍스트 품질 향상 (투표 시스템 구현)
- `surya-ocr` 0.16.0 사용 (0.17.1은 transformers 5.x 호환 문제)
