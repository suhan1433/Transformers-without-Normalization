## Transformers-without-Normalization
트랜스포머 학습 속도 향상을 위해 Layer Norm의 위치 변경 및 Dynamic Tanh으로 교체 실험

### 개요
트랜스포머 모델의 학습 효율성을 향상시키기 위해, 기존 LayerNorm의 계산 복잡도 및 병렬화 제약 문제를 Dynamic Tanh (DyT) 정규화 기법으로 대체하는 실험을 수행했습니다. 
LayerNorm의 통계량 계산 오버헤드와 GPU 동기화 비용을 element-wise 연산 기반의 DyT로 해결하여, 모델 성능 유지와 동시에 훈련 속도 개선을 목표로 합니다.
[https://arxiv.org/abs/2503.1062](https://arxiv.org/abs/2503.10622)

약 2만 7천 개의 영어-독일어 번역 데이터셋을 활용하여, Loss 수렴 속도와 BLEU 점수를 평가했습니다.

## Transformer 정규화 방식 비교

| 구분 | PostLayerNorm (Vanilla) | PreLayerNorm | Dynamic Tanh (DyT) |
|------|-------------------------|--------------|-------------------|
| **구조** | LayerNorm → Residual Connection | Residual Connection → LayerNorm | LayerNorm을 DyT(`Tanh(αx)`)로 대체 |
| **제안 배경** | Transformer 원본 구조 | • PostLayerNorm의 학습 불안정성 해결<br>• Warm-up 의존성 문제 개선 | • LayerNorm의 연산 비용 문제<br>• 평균/분산 계산으로 인한 병목 현상 |
| **주요 장점** | • 원본 설계 그대로 사용 | • Warm-up 없이도 안정적 학습<br>• 기울기 소실/폭주 완화<br>• 빠른 수렴 속도 | • LayerNorm보다 빠른 연산<br>• 메모리 효율성 향상<br>• 학습 가능한 파라미터 α로 적응적 정규화 |
| **주요 단점** | • 긴 Warm-up 필요<br>• 학습 초기 불안정<br>• Gradient explosion 위험 | •  | • 상대적으로 새로운 기법<br>• 검증 사례가 제한적 |
| **현재 상태** | 초기 Transformer 모델에서 사용 | 현재 대부분 모델의 표준 | 연구 단계의 새로운 접근법 |

## LayerNorm vs DyT
### LayerNorm의 동작 방식
```
LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
```
- **μ, σ²**: 입력의 평균, 분산 (전체 feature에 대해 계산 필요)
- **γ, β**: 학습 가능한 스케일링/시프트 파라미터

<img width="323" height="350" alt="스크린샷 2025-07-24 오후 5 11 21" src="https://github.com/user-attachments/assets/044ccc1c-3e74-4837-9f4f-cd5853a7e3b5" />

### 주요 문제점
1. **순차적 계산**: 평균 → 분산 → 정규화 단계별 진행 (각 단계가 끝나야 다음 단계 가능하여 속도 저하)
2. **전역 의존성**: 각 요소의 정규화는 전체 임베딩 차원(feature)값들의 평균과 분산에 의존 (모든 Thread가 데이터 공유해야 각 요소의 평균/분산 계산 가능)
3. **GPU 동기화**: 통계량 계산 시 메모리 동기화 필요하여 병목 현상 (Block내 모든 Thread가 값을 공유해야 함)
   
→ 위와 같은 이유로 Layernorm은 GPU에서 완전한 병렬화가 불가능

---
GPU 동기화 과정 예시
1. 연산 단위 할당
한 개의 토큰 임베딩(예: 512차원)에 대해 LayerNorm 연산을 수행할 때,
GPU의 한 Block이 해당 토큰 임베딩 전체를 담당합니다.
임베딩 차원(예: 512)을 Warp(32 threads) 단위로 나누어 각 Warp가 일부 차원을 처리합니다.
예: 512차원 → 16개의 Warp(각 32차원씩)
2. 평균 및 분산 계산 과정
각 Thread는 자신이 맡은 차원의 값을 처리합니다.
평균 및 분산 계산을 위해 Block 내의 모든 Thread가 값을 Shared Memory에 모아 합산합니다.
합산 결과를 Block 내 모든 Thread가 공유해야 하므로,
Thread 동기화(Synchronization)가 필요합니다.
3. 정규화 및 결과 전달
계산된 평균/분산을 이용해 각 Thread가 자신이 맡은 차원을 정규화합니다.
최종 결과는 다시 각 Thread를 통해 출력됩니다.

- SM (Streaming Multiprocessor):
GPU의 연산 유닛. 여러 Block을 동시에 실행할 수 있음.
- Kernel:
GPU에서 실행되는 함수(연산 단위). Grid 단위로 실행됨.
- Grid:
여러 Block으로 구성된 Kernel 실행 단위.
- Block:
여러 Thread로 구성. 한 Block이 한 토큰 임베딩을 담당.
- Warp:
32개의 Thread로 구성된 실행 단위. Block 내에서 차원별로 분배됨.

<img width="906" height="366" alt="스크린샷 2025-07-24 오후 5 37 00" src="https://github.com/user-attachments/assets/bd7aefe2-1d34-447d-adf2-ae54cf17f081" />
<img width="438" height="555" alt="스크린샷 2025-07-24 오후 5 45 21" src="https://github.com/user-attachments/assets/e1854b82-1880-4c85-a1a1-4eedf220d24c" />


- 요약
  
LayerNorm 연산 시,
한 Block이 한 토큰 임베딩을 담당하고,
여러 Warp가 임베딩 차원을 분할 처리합니다.
또한 평균/분산 계산을 위해 Block 내 Thread 동기화가 필요합니다.
https://github.com/NVIDIA/apex/blob/master/csrc/layer_norm_cuda_kernel.cu : 과정


### DyT의 해결책
```
DyT(x) = γ * tanh(α * x) + β
```
- **Element-wise 독립 연산**: 각 요소를 병렬로 동시 처리(모든 Thread가 각자 맡은 값만 처리)
- **통계량 계산 불필요**: 평균/분산 계산 과정 제거
- **완전한 병렬화**: GPU Thread(코어) 간 동기화 없이 처리

DyT의 경우 
DyT의 경우 각 임베딩 값에 대해 학습 가능한 α를 곱해주기만 하면 되기에 독립적인 연산이 가능합니다.
즉 평균 및 분산 계산을 위해 임베딩의 각 값을 Thread 간 동기화가 필요없습니다.


```python
import torch
import torch.nn as nn

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias
```

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

DyT의 경우 S곡선이 뚜렷하게 나오며, LayerNorm의 경우 S곡선이 애매하다

LayerNorm Layer 개수 (총 30개)

Encoder (총 12개)
- 각 블록마다 LayerNorm 2개:
  - Multi-Head Self-Attention 뒤 1개
  - Feed-Forward Network 뒤 1개
- 총 6개 블록 → 2 × 6 = 12개

Decoder (총 18개)
- 각 블록마다 LayerNorm 3개:
  - Masked Multi-Head Self-Attention 뒤 1개
  - Multi-Head Cross-Attention 뒤 1개
  - Feed-Forward Network 뒤 1개
- 총 6개 블록 → 3 × 6 = 18개
  

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
  - DyT는 LayerNorm과 **완전히 다른 파라미터 구조**를 가짐:
    - LayerNorm: `γ * (정규화) + β` 
    - DyT: `γ * tanh(α * x) + β` 
  - Pre-trained 모델의 Fine-tuning 시:
    - 새로운 `α` 파라미터 초기화 및 모든 파라미터(`α`, `γ`, `β`) 재조정 필요
    - 아키텍처 변경으로 인한 추가적인 학습 비용 발생
  - Inference 적용 시에도 동일한 문제

## Appendix
- Post일 때 기울기 소실 문제

- Pre일 때 기울기 폭발 문제
Pre-LN을 역전파 과정을 보면

* x → [LayerNorm → Sublayer] → (+x) → output
* ∂L/∂x = ∂L/∂output · ∂output/∂x
* output = x + sublayer(LayerNorm(x))


### Reference
[https://arxiv.org/abs/2503.1062](https://arxiv.org/abs/2503.10622) : Transformers without Normalization

https://arxiv.org/pdf/2002.04745 : On Layer Normalization in the Transformer Architecture

https://julianhatzky.me/blog/2025/tanh/?utm_source=chatgpt.com

GPU 구조 
https://xoft.tistory.com/75 ,
https://cuda-programming.blogspot.com/2013/01/thread-and-block-heuristics-in-cuda.html ,
https://dlsys.cs.washington.edu/pdf/lecture5.pdf,

