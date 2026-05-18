# policies 설명서

이 폴더는 "로봇이 무엇을 보고, 무엇을 읽고, 어떻게 움직일지 정하는 뇌"에 해당합니다.

## 파일 역할

| 파일 | 역할 |
|---|---|
| `act.py` | ACT 정책입니다. 이미지, 언어, 현재 state를 받아 CVAE latent와 Transformer로 미래 action chunk를 예측합니다. |
| `pi05.py` | pi0.5 스타일 정책입니다. VLM 구조와 flow matching 행동 생성을 흉내 냅니다. |
| `config.py` | `ACTConfig`, `Pi05Config`, `PolicyFactory`를 한곳에서 만들고 검증합니다. |
| `__init__.py` | 외부에서 `from policies import ACTPolicy`처럼 쉽게 가져오게 해줍니다. |

## 공통 입력 흐름

학습할 때 batch는 대략 이렇게 생깁니다.

```python
batch = {
    "images": Tensor,        # (B, num_cameras, 3, H, W)
    "language_ids": Tensor,  # (B, text_len)
    "state": Tensor,         # (B, state_dim)
    "actions": Tensor,       # (B, chunk_size, action_dim)
    "action_mask": Tensor,   # (B, chunk_size), True면 padding이라 loss에서 무시
}
```

각 정보의 뜻은 다음과 같습니다.

| 이름 | 쉬운 뜻 | 로봇에게 주는 정보 |
|---|---|---|
| `images` | 눈 | 물체가 어디 있는지, 장면이 어떤지 |
| `language_ids` | 명령 쪽지 | 사람이 무엇을 하라고 했는지 |
| `state` | 몸 상태 | 지금 팔과 그리퍼가 어디 있는지 |
| `actions` | 정답 움직임 | 사람이 시범으로 보여준 미래 움직임 |
| `action_mask` | 빈칸 표시 | 짧은 episode를 padding했을 때 어느 칸을 무시할지 |

## ACT forward 흐름

호출 모양은 다음과 같습니다.

```python
pred_actions = act_policy(images, language_ids, state)
```

반환값은 `(B, action_chunk_size, action_dim)`입니다.

### 단계별 설명

1. `ResNetImageEncoder`

   여러 카메라 이미지를 ResNet으로 읽고 spatial token으로 바꿉니다.

   ```text
   images
       (B, num_cameras, 3, H, W)
       -> 각 카메라를 ResNet에 통과
       -> feature map projection
       -> (B, num_cameras * h * w, hidden_dim)
   ```

   전달되는 정보는 "어디에 물체가 있고 장면이 어떻게 생겼는가"입니다.

2. `LanguageEncoder`

   명령 문장을 token id로 받은 뒤 Transformer encoder로 읽습니다.

   ```text
   language_ids
       (B, text_len)
       -> token embedding
       -> positional encoding
       -> Transformer encoder
       -> (B, hidden_dim)
   ```

   전달되는 정보는 "무엇을 해야 하는가"입니다.

3. latent, state, language, image token 결합

   ACT는 학습 중 정답 action chunk를 보고 CVAE latent를 만듭니다. 추론 중에는 latent를 0으로 두고 행동을 예측합니다.

   ```text
   training:
       [cls | state | action chunk]
       -> VAE encoder
       -> mu, logvar
       -> latent sample

   inference:
       latent = zeros
   ```

   이어서 latent, state, language, image spatial token을 한 sequence로 묶어 Transformer encoder에 넣습니다.

   ```text
   [latent token | state token | language token | image spatial tokens]
       -> Transformer encoder
       -> memory tokens
   ```

4. action query 생성

   `action_chunk_size`개 만큼 learnable query를 만듭니다.

   ```text
   action_query
       (B, chunk_size, hidden_dim)
   ```

   각 query는 "미래의 1번째 움직임", "미래의 2번째 움직임" 같은 자리표입니다.

5. Transformer decoder

   ```text
   action query
   memory tokens
       -> self attention
       -> cross attention
       -> feed-forward
   ```

   action query들이 장면, 명령, 현재 state, latent 정보를 cross-attention으로 봅니다.

6. action head

   마지막으로 각 decoder output을 실제 로봇 action 숫자로 바꿉니다.

   ```text
   decoder output
       -> Linear
       -> (B, chunk_size, action_dim)
   ```

### ACT loss와 추론

학습에서는 `compute_loss(...)`가 masked L1 reconstruction loss와 KL loss를 함께 계산합니다.

```python
loss, metrics = act_policy.compute_loss(
    images,
    language_ids,
    state,
    actions,
    action_mask,
)
```

```text
loss = masked_l1(pred_actions, actions) + kl_weight * KL(q(z | state, actions) || N(0, I))
```

추론에서는 `predict_action(...)` 또는 `select_action(...)`을 사용합니다. 기본은 `n_action_steps`만큼 action queue를 채워서 여러 step 동안 재사용하고, `temporal_ensemble_coeff`가 설정되어 있으면 매 step 새 chunk를 예측해 temporal ensemble을 계산합니다.

### ACT에서 논문 대비 빠진 로직

현재 구현은 ACT 논문과 LeRobot 구현의 주요 학습 구조를 반영하지만, LeRobot 전체 스택을 그대로 가져온 것은 아닙니다.

| 논문 로직 | 현재 코드 상태 |
|---|---|
| action chunk 예측 | 있음 |
| multi-camera 이미지 사용 | spatial token 방식으로 있음 |
| 현재 로봇 state 조건화 | 있음 |
| CVAE latent variable | 있음 |
| KL loss | 있음 |
| action queue | 있음 |
| temporal ensemble | optional로 있음 |
| LeRobot Hub dataset/processor 전체 호환 | 아직 아님 |
| ALOHA 원본 episode 처리와 공식 평가 스택 | 아직 아님 |

## pi0.5 forward 흐름

일반 forward는 다음처럼 호출됩니다.

```python
pred_actions = pi05_policy(images, language_ids)
```

학습에서는 flow matching loss를 사용합니다.

```python
loss = pi05_policy.compute_flow_matching_loss(
    images,
    language_ids,
    ground_truth_actions,
)
```

### 일반 forward 단계

1. 이미지 입력

   ```text
   images
       (B, num_cameras, 3, H, W)
       -> 카메라별로 vision encoder 통과
   ```

2. `SigLIPVisionEncoder`

   이미지를 patch로 나누고 ViT 스타일 encoder에 넣습니다.

   ```text
   image
       -> patch embedding
       -> transformer layers
       -> patch_features
   ```

   전달되는 정보는 "장면의 시각적 토큰들"입니다.

3. `MultimodalProjector`

   vision feature 차원을 LLM 차원으로 맞춥니다.

   ```text
   vision_width
       -> llm_width
   ```

   이것은 "그림 언어"를 "LLM이 읽을 수 있는 언어"로 번역하는 단계입니다.

4. `LLaMADecoder`

   명령 문장 token을 LLaMA 스타일 decoder에 넣습니다.

   ```text
   language_ids
       -> token embedding
       -> decoder layers
       -> text_features
   ```

5. multimodal attention

   image token, text token, action pool token을 이어 붙입니다.

   ```text
   [vision tokens | text tokens | action_pool]
       -> multimodal attention
       -> action_token
   ```

   `action_token`은 "이 장면에서 이 명령을 수행하기 위한 요약 정보"입니다.

6. `ActionExpert`

   `action_token`을 미래 행동 chunk로 바꿉니다.

   ```text
   action_token
       -> MLP
       -> (B, chunk_size, action_dim)
   ```

## pi0.5 flow matching 흐름

flow matching은 정답 action을 바로 맞히는 대신, 노이즈에서 정답 action으로 가는 방향을 배우는 방식입니다.

학습 중에는 다음이 일어납니다.

1. 정답 action을 `x0`라고 부릅니다.
2. 랜덤 노이즈 `eps`를 만듭니다.
3. 시간 `t`를 0과 1 사이에서 고릅니다.
4. 중간 action `x_t`를 만듭니다.

```text
x_t = (1 - t) * eps + t * x0
```

5. 모델은 `x_t`에서 `x0` 쪽으로 가는 속도 `v`를 맞힙니다.

```text
target_velocity = x0 - eps
pred_velocity = model(x_t, t, images, language_ids)
loss = MSE(pred_velocity, target_velocity)
```

현재 코드에서 `_predict_velocity(...)`가 이 역할을 합니다.

```python
velocity = policy._predict_velocity(
    x_t,
    t,
    images,
    language_ids,
)
```

여기서 forward로 전달되는 정보는 다음과 같습니다.

| 정보 | 의미 |
|---|---|
| `x_t` | 아직 완성되지 않은 중간 action |
| `t` | 지금이 노이즈에서 정답으로 가는 과정 중 몇 번째 위치인지 |
| `images` | 현재 장면 |
| `language_ids` | 수행할 명령 |

## pi0.5에서 논문 대비 빠진 로직

현재 `pi05.py`는 논문 구조를 공부하기 위한 단순화 버전입니다.

| 논문 로직 | 현재 코드 상태 |
|---|---|
| flow matching objective | 있음 |
| action expert | 있음 |
| SigLIP/LLaMA 스타일 모듈 | 직접 구현한 유사 구조 |
| PaliGemma 3B 사전학습 가중치 | 없음 |
| 실제 PaliGemma tokenizer | 없음 |
| 대규모 co-training 데이터 | 없음 |
| semantic subtask, object detection 등 high-level 예측 | 없음 |
| 논문 수준 평가/벤치마크 | 없음 |

## 디버깅할 때 먼저 볼 shape

ACT가 터지면 다음 shape부터 확인하세요.

```text
images        (B, 3, 3, 224, 224)
language_ids  (B, 128)
state         (B, 28)
actions       (B, 32, 14)
prediction    (B, 32, 14)
```

위 예시는 작은 디버깅 설정입니다. 현재 ACT 기본값은 LeRobot 쪽에 맞춰 `chunk_size=100`, `action_dim=14`, `state_dim=28`입니다. 빠른 smoke test나 작은 GPU에서는 `--chunk-size 16 --n-action-steps 4`처럼 줄여서 확인할 수 있습니다.

pi0.5가 터지면 다음 shape부터 확인하세요.

```text
images        (B, 3, 3, 224, 224)
language_ids  (B, 128)
actions       (B, 32, 14)
x_t           (B, 32, 14)
velocity      (B, 32, 14)
```

## 어떤 정책부터 봐야 하나

처음 읽는다면 이 순서가 가장 편합니다.

1. `ACTConfig`
2. `ACTPolicy.forward`
3. `ACTLoss`
4. `train.py`의 `train_one_epoch`
5. `Pi05Config`
6. `Pi05Policy.compute_flow_matching_loss`
7. `Pi05Policy._predict_velocity`

ACT를 먼저 이해하면 pi0.5도 훨씬 쉽게 읽힙니다. ACT는 "정답 행동을 바로 맞히기"이고, pi0.5는 "노이즈를 행동으로 바꾸는 방향을 배우기"입니다.
