# Mistral-7B Fine-tuning Experiment

## Overview
이 실험은 Mistral-7B 모델을 YouTube 비디오 스크립트의 핵심 장면 추출을 위해 fine-tuning한 실험입니다.

## 실험 구성

### Quantization
- 모델의 메모리 사용량을 줄이고 추론 속도를 향상시키기 위해 4bit, 8bit 양자화 적용

### Fine-tuning 설정
- LoRA를 적용하여 4bit 모델 fine-tuning 진행
- 목표: 실제 "가장 많이 다시 본 장면"의 특징을 학습하여 모델이 예측하도록 하는 것
- 특히 "요약 결과가 가장 많이 다시 본 장면이 될 수 있느냐"에 대한 문제 해결 목적

### 데이터 처리
- Chunk 기반 generation
  - Time interval 형식: ex) [[0.5,5.2],[30.2,35.1]]
  - Time Retriever 과정 없이 해당 시점 발화 추출 가능
  - 문장을 생성하지 않고 Time Interval 형식의 답변을 생성하여 Inference 속도 감소

### Training Process
- Script를 1024 토큰 단위로 chunk 분할
- 각 chunk별 "가장 많이 다시 본 장면" 존재 여부 확인
- 없는 경우 "No Answer" 레이블 설정

### Inference
- 각 Chunk별 생성 결과와 Time Interval을 60초 단위로 묶음
- 중복 및 기존도로 앞 뒤 30초, 총 60초 구간 n개 생성

### Evaluation
- 생성된 n개의 구간을 정답으로 가정
- 기존 Metric과 완성으로 결과 측정

## 실험 결과

| Category      | X    | O    |
|--------------|------|------|
| Entertainment | 0.32 | 0.27 |
| Comedy       | 0.23 | 0.25 |
| Science      | 0.29 | 0.31 |
| Education    | 0.39 | 0.27 |
| News         | 0.38 | 0.28 |
| 평균          | 0.32 | 0.28 |

## 결론
- 약 50% 추론 시간 감소 효과가 있지만 Fine-Tuning을 하지 않은 모델에 비해 대체로 낮은 성능
- Fine-tuning을 통해 기존에는 time stamp와 text를 모두 생성하도록 하였지만 time stamp만 생성하도록 하여 출력 포맷을 변경하여 추론 시간을 줄이고자 함
- 출력 포맷 변경으로 시각상 Vision modal과의 결합이 진행되지 못함
- 현재는 Base Model을 사용하지만 더 많은 데이터로 fine-tuning 진행시 더 나은 성능 기대 