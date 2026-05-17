# policies 설명서

이 폴더는 "로봇이 무엇을 보고, 무엇을 읽고, 어떻게 움직일지 정하는 뇌"에 해당합니다.

## 파일 역할

| 파일 | 역할 |
|---|---|
| `act.py` | ACT 정책입니다. 이미지, 언어, 현재 state를 받아 미래 action chunk를 바로 예측합니다. |
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

   여러 카메라 이미지를 ResNet으로 읽습니다.

   ```text
   images
       (B, num_cameras, 3, H, W)
       -> 각 카메라를 ResNet에 통과
       -> (B, num_cameras * hidden_dim)
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

3. image feature와 language feature 결합

   ```text
   visual_feats + lang_feats
       -> concat
       -> Linear + LayerNorm + GELU
       -> fused context
   ```

   여기서 로봇은 "현재 장면에서 이 명령을 수행해야 한다"는 하나의 context를 갖게 됩니다.

4. action token 생성

   `action_chunk_size`개 만큼 learnable token을 만듭니다.

   ```text
   action_tokens
       (B, chunk_size, hidden_dim)
   ```

   각 token은 "미래의 1번째 움직임", "미래의 2번째 움직임" 같은 자리표입니다.

5. state FiLM 주입

   현재 로봇 state를 `gamma`, `beta`로 바꿔 Transformer decoder layer마다 넣습니다.

   ```text
   state
       -> Linear
       -> gamma, beta
       -> decoder 내부 feature 조절
   ```

   전달되는 정보는 "지금 팔이 여기 있으니, 이 자세에서 시작해야 한다"입니다.

6. Transformer decoder

   action token들이 fused context를 cross-attention으로 봅니다.

   ```text
   action tokens query
   fused context key/value
       -> self attention
       -> cross attention
       -> feed-forward
   ```

7. action head

   마지막으로 각 token을 실제 로봇 action 숫자로 바꿉니다.

   ```text
   decoder output
       -> MLP
       -> (B, chunk_size, action_dim)
   ```

### ACT에서 논문 대비 빠진 로직

현재 구현은 ACT 논문의 핵심인 action chunk 아이디어는 담고 있지만, 원 논문 전체를 그대로 재현하지는 않습니다.

| 논문 로직 | 현재 코드 상태 |
|---|---|
| action chunk 예측 | 있음 |
| multi-camera 이미지 사용 | 있음 |
| 현재 로봇 state 조건화 | 있음 |
| CVAE latent variable | 없음 |
| KL loss | 없음 |
| temporal ensemble | 없음 |
| ALOHA 원본 episode 처리와 완전 동일한 loader | 아님 |

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
