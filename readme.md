## Transformers-without-Normalization
트랜스포머 학습 속도 향상을 위해 Layer Norm의 위치 변경 및 Dynamic Tanh으로 교체 실험

### 개요
트랜스포머의 학습 속도를 향상시키기 위해, LayerNorm 연산에서 발생하는 병목 현상을 Dynamic Tanh로 교체하여 해결하는 실험을 진행했습니다.

약 2만 7천 개의 영어-독일어 번역 데이터셋을 활용하여, Loss 수렴 속도와 BLEU 점수를 평가했습니다.

### Transformer 정규화 방식 비교

| 구분 | PostLayerNorm (Vanilla) | PreLayerNorm | Dynamic Tanh (DyT) |
|------|-------------------------|--------------|-------------------|
| **구조** | LayerNorm → Residual Connection | Residual Connection → LayerNorm | LayerNorm을 DyT(`Tanh(αx)`)로 대체 |
| **제안 배경** | Transformer 원본 구조 | • PostLayerNorm의 학습 불안정성 해결<br>• Warm-up 의존성 문제 개선 | • LayerNorm의 연산 비용 문제<br>• 평균/분산 계산으로 인한 병목 현상 |
| **주요 장점** | • 원본 설계 그대로 사용 | • Warm-up 없이도 안정적 학습<br>• 기울기 소실/폭주 완화<br>• 빠른 수렴 속도 | • LayerNorm보다 빠른 연산<br>• 메모리 효율성 향상<br>• 학습 가능한 파라미터 α로 적응적 정규화 |
| **주요 단점** | • 긴 Warm-up 필요<br>• 학습 초기 불안정<br>• Gradient explosion 위험 | • 기존 PreNorm 모델과의 호환성 | • 상대적으로 새로운 기법<br>• 검증 사례가 제한적 |
| **현재 상태** | 초기 Transformer 모델에서 사용 | 현재 대부분 모델의 표준 | 연구 단계의 새로운 접근법 |

### 방법론
Dynamic tanh: 단순한 element-wise 연산 → 병렬화 쉬움
LayerNorm: 평균/분산 계산 필요 → 계산 비용 높음

## 실험 

### Dataset

**Multi30K** (영어-독일어 번역 데이터셋)
- **Train**: 29,000
- **Validation**: 1,014  
- **Test**: 1,000

### 학습 파라미터
| 항목              | 값                                 |
|------------------|------------------------------------|
| `hidden_size`    | `512`                              |
| `Multihead_splits` | `8`                              |
| `enc_blocks`     | `6`                                |
| `dec_blocks`     | `6`                                |
| `dropout_p`      | `0.1`                              |
| `max_length`     | `128`                              |
| `batch_size`     | `150`                              |
| `epochs`         | `200`                              |
| `patience`       | `3`                                |
| `Scheduler`      | `CosineAnnealingLR`               |
| `활성화 함수`     | `ReLU`                             |
| `Optimizer`      | `Adam(모멘텀=0.9, β2=0.98, eps=1e-9)` |
| `warmup`       | `False` |


### PostLayerNorm(Vanila)
<img width="224" height="501" alt="스크린샷 2025-07-22 오후 12 33 26" src="https://github.com/user-attachments/assets/e04d3f19-48c5-4e9c-8db8-fd6cec6d6389" />

```python
        self.attn = MultiHead(hidden_size, n_splits)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout_p)

        # feedforward
        self.fc = nn.Sequential(
            # 512 * 4 = 2048
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc_dropout = nn.Dropout(dropout_p)
    
    def forward(self, x, mask):
        # Post-LN
        z = self.attn_norm(x + self.attn_dropout(self.attn(Q=x, K=x, V=x, mask = mask)))
        z = self.fc_norm(z + self.fc_dropout(self.fc(z)))

        return z, mask
```

### PreLayerNorm
<img width="219" height="473" alt="스크린샷 2025-07-22 오후 12 34 36" src="https://github.com/user-attachments/assets/52b35e65-47db-4033-9be8-33f3bb32c8e4" />

```python
        self.attn = MultiHead(hidden_size, n_splits)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout_p)

        # feedforward
        self.fc = nn.Sequential(
            # 512 * 4 = 2048
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc_dropout = nn.Dropout(dropout_p)
    def forward(self, x, mask):        
        # Pre-LN
        z = self.attn_norm(x)
        z = x + self.attn_dropout(self.attn(Q=z, K=z, V=z, mask = mask))
        z = z + self.fc_dropout(self.fc(self.fc_norm(z)))

        return z, mask
```

### DyT(Dynamic Tanh)
<img width="337" height="194" alt="스크린샷 2025-07-22 오후 12 37 30" src="https://github.com/user-attachments/assets/3e5523db-3d7e-480b-aa54-520cb2036e6c" />

```python
        self.attn = MultiHead(hidden_size, n_splits)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout_p)

        # feedforward
        self.fc = nn.Sequential(
            # 512 * 4 = 2048
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, mask):
        # DyT
        z = self.attn_dyt(x)
        z = x + self.attn_dropout(self.attn(Q=z, K=z, V=z, mask = mask))
        z = z + self.fc_dropout(self.fc(self.fc_dyt(z)))

        return z, mask
```


### PreLayerNorm과 DyT의 Layer 형상 비교
<img width="857" height="586" alt="스크린샷 2025-07-22 오후 12 40 38" src="https://github.com/user-attachments/assets/d1e51137-a46d-4fb9-9697-218e35206c93" />

## 결과
<img width="1591" height="541" alt="스크린샷 2025-07-22 오후 12 52 09" src="https://github.com/user-attachments/assets/16577522-454e-47ff-8a74-dc5c14f1dc37" />

* 모델 성능 비교

| 모델            | Train Loss ↓ | Val Loss ↓ | BLEU ↑  | Epoch | 수렴 속도 |
|-----------------|--------------|------------|---------|-------|-----------|
| PostLayerNorm   | 5.74         | 10.2       | 0.00    | 4     | 매우 느림  |  
| PreLayerNorm    | 0.55         | 2.1        | 0.26    | 18    | 보통      |
| DyT_Decoder     | 0.87         | 1.74       | 0.31    | 9     | ⚡ 빠름   |
| DyT_Encoder     | 0.76         | 1.78       | 0.29    | 9     | ⚡ 빠름   |
| DyT             | 0.85         | 2.4        | 0.22    | 22    | 느림      |

모델 설명:
- PostLayerNorm : LayerNorm → Residual Connection
- PreLayerNorm : Residual Connection → LayerNorm
- DyT_Decoder : PreLayerNorm 구조에서 디코더의 LayerNorm만 DyT로 변경
- DyT_Encoder : PreLayerNorm 구조에서 인코더의 LayerNorm만 DyT로 변경
- DyT : PreLayerNorm 구조에서 인코더, 디코더 LayerNorm을 전부 DyT로 변경

결과:
- **DyT_Decoder**가 전체적으로 가장 우수한 성능을 보임 (Val Loss: 1.74, BLEU: 0.31)
- **DyT_Encoder**와 **DyT_Decoder** 모두 9 epoch만에 우수한 성능에 도달하여 빠른 수렴 속도를 보임
- PreLayerNorm 대비 DyT 계열 모델들은 2배 빠른 수렴 속도 달성
- PostLayerNorm은 Warmup과정이 없기에 학습이 제대로 이루어지지 않음 (BLEU: 0.00)

## 한계점

- 대규모 모델에서의 성능 문제
  - 여러 아티클을 검토한 결과, 대규모 모델에서는 DyT의 성능이 LayerNorm 대비 오히려 저하되는 경우가 확인됨
  - 현재까지 검증된 사례가 제한적이어서 실제 프로덕션 환경 적용에는 검토 필요

- Fine-tuning 및 Inference 적용의 제약
  - DyT는 `tanh(ax)`에서 스케일링 파라미터 `a`를 학습해야 하는 구조
  - Pre-trained 모델의 Fine-tuning 시:
    - 기존 LayerNorm 가중치와 호환되지 않아 아키텍처 변경 필요
    - `a` 파라미터 초기화 및 재학습으로 인한 추가적인 학습 비용 발생
    - Inference할때도 동일한 문제
  


### Reference
[https://arxiv.org/abs/2503.1062](https://arxiv.org/abs/2503.10622) : Transformers without Normalization

https://arxiv.org/pdf/2002.04745 : On Layer Normalization in the Transformer Architecture
