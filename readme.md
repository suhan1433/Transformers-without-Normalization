## Transformers-without-Normalization
트랜스포머 학습 속도 향상을 위해 Layer Norm의 위치 변경 및 Dynamic Tanh으로 교체 실험

### 개요
트랜스포머의 학습 속도를 높이기 위해 Layer Norm의 연산의 병목 현상을 해결하기 위해 Dynamic Tanh 변경 실험.

약 2만7천개의 영어-독일 번역 데이터셋을 활용하여, Loss의 수렴 속도와 Bleu 점수 확인  

### Transformer 정규화 방식 비교

| 구분 | PostLayerNorm (Vanilla) | PreLayerNorm | Dynamic Tanh (DyT) |
|------|-------------------------|--------------|-------------------|
| **구조** | LayerNorm → Residual Connection | Residual Connection → LayerNorm | LayerNorm을 DyT(`Tanh(αx)`)로 대체 |
| **제안 배경** | Transformer 원본 구조 | • PostLayerNorm의 학습 불안정성 해결<br>• Warm-up 의존성 문제 개선 | • LayerNorm의 연산 비용 문제<br>• 평균/분산 계산으로 인한 병목 현상 |
| **주요 장점** | • 원본 설계 그대로 사용 | • Warm-up 없이도 안정적 학습<br>• 기울기 소실/폭주 완화<br>• 빠른 수렴 속도 | • LayerNorm보다 빠른 연산<br>• 메모리 효율성 향상<br>• 학습 가능한 파라미터 α로 적응적 정규화 |
| **주요 단점** | • 긴 Warm-up 필요<br>• 학습 초기 불안정<br>• Gradient explosion 위험 | • 기존 PreNorm 모델과의 호환성 | • 상대적으로 새로운 기법<br>• 검증 사례가 제한적 |
| **현재 상태** | 초기 Transformer 모델에서 사용 | 현재 대부분 모델의 표준 | 연구 단계의 새로운 접근법 |

### Dataset

**Multi30K** (영어-독일어 번역 데이터셋)
- **Train**: 29,000
- **Validation**: 1,014  
- **Test**: 1,000








### Reference
[https://arxiv.org/abs/2503.1062](https://arxiv.org/abs/2503.10622) : Transformers without Normalization

https://arxiv.org/pdf/2002.04745 : On Layer Normalization in the Transformer Architecture
