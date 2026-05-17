# bh-VLA

bh-VLA는 로봇에게 일을 가르치기 위한 작은 Vision-Language-Action 학습 프레임워크입니다.

아주 쉽게 말하면 다음 순서로 움직입니다.

1. 로봇이 카메라로 현재 상황을 봅니다.
2. 사람이 쓴 명령 문장을 읽습니다. 예: "빨간 컵을 집어"
3. 현재 로봇 관절 상태를 같이 봅니다.
4. 앞으로 몇 순간 동안 팔을 어떻게 움직일지 예측합니다.
5. 예측한 첫 움직임을 로봇에 보냅니다.

이 저장소는 두 종류의 정책을 제공합니다.

| 정책 | 파일 | 한 줄 설명 | 현재 상태 |
|---|---|---|---|
| ACT | `policies/act.py` | 이미지를 보고 여러 개의 미래 행동을 한 번에 예측합니다. | 실행 가능한 경량 구현 |
| pi0.5 | `policies/pi05.py` | 큰 VLM과 flow matching 방식의 행동 생성을 흉내 냅니다. | 연구용 스캐폴드 |

중요합니다. 이 코드는 논문을 완전히 재현한 공식 구현이 아닙니다. 특히 pi0.5는 PaliGemma 사전학습 가중치와 실제 tokenizer, 대규모 co-training 데이터가 없기 때문에 논문 성능을 기대하면 안 됩니다. 현재 목적은 구조를 이해하고 작은 데이터로 실험하는 것입니다.

## 논문 기준 검토

| 논문 | 코드에 들어간 핵심 아이디어 | 아직 빠진 부분 |
|---|---|---|
| ACT, arXiv:2304.13705 | action chunk 예측, multi-camera image encoder, Transformer decoder, 현재 state 조건부 행동 예측 | 원 논문의 CVAE latent, KL loss, temporal ensemble, ALOHA 원본 데이터 파이프라인 |
| pi0.5, arXiv:2504.16054 | SigLIP 스타일 vision encoder, LLaMA 스타일 decoder, action expert, flow matching loss와 sampling 흐름 | 실제 PaliGemma 3B 가중치, 실제 tokenizer, 대규모 heterogeneous co-training, semantic subtask/object prediction, 논문 수준의 평가 코드 |

따라서 이 저장소를 읽을 때는 이렇게 보면 좋습니다.

- ACT는 "작은 imitation learning baseline"으로 사용할 수 있습니다.
- pi0.5는 "논문 구조를 따라가며 배우는 뼈대"에 가깝습니다.
- 논문과 같은 성능을 내려면 빠진 항목을 추가해야 합니다.

## 설치

Python 3.10 이상을 권장합니다.

```bash
git clone https://github.com/sphill9803/bh-vla.git
cd bh-vla

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

GPU가 없으면 ACT는 CPU에서도 구조 확인이 가능합니다. pi0.5 기본 설정은 매우 큽니다. 작은 GPU나 CPU에서는 설정을 줄여야 합니다.

## 데이터 모으기

로봇 없이 먼저 테스트 데이터를 만들 수 있습니다.

```bash
python collect_data.py --mode act --task-name demo --num-episodes 3 --skip-robot
```

그러면 보통 아래처럼 저장됩니다.

```text
data/demo/
├── dataset_info.json
├── episode_0000/
│   ├── images.npy
│   ├── states.npy
│   ├── actions.npy
│   └── metadata.json
└── episode_0001/
    └── ...
```

각 파일의 뜻은 다음과 같습니다.

| 파일 | 뜻 |
|---|---|
| `images.npy` | 카메라 이미지들입니다. 모양은 대략 `(시간, 높이, 너비, 3)`입니다. |
| `states.npy` | 그 순간 로봇의 관절 상태입니다. |
| `actions.npy` | 사람이 시범으로 보여준 다음 움직임입니다. |
| `metadata.json` | 명령 문장, 프레임 수, 수집 시간 같은 설명입니다. |

## 학습

처음에는 ACT부터 권장합니다.

```bash
python train.py --mode act --data-dir ./data/demo --device cpu --epochs 2 --batch-size 2
```

GPU가 있다면 다음처럼 실행할 수 있습니다.

```bash
python train.py --mode act --data-dir ./data/demo --device cuda
```

pi0.5는 훨씬 무겁습니다.

```bash
python train.py --mode pi05 --data-dir ./data/demo --device cuda --batch-size 1
```

학습이 끝나면 checkpoint가 저장됩니다.

```text
checkpoints/
├── act_best.pt
├── act_last.pt
└── ...
```

## 추론

로봇 없이 checkpoint가 열리는지 확인합니다.

```bash
python inference.py --mode act --checkpoint ./checkpoints/act_last.pt --test-mode --device cpu
```

실제 로봇에서 실행할 때는 다음처럼 사용합니다.

```bash
python inference.py --mode act --checkpoint ./checkpoints/act_last.pt --language "빨간 컵을 집어"
```

## 전체 코드 흐름

```text
collect_data.py
    사람이 시범 데이터를 모음
    아래 파일들을 저장
        images.npy
        states.npy
        actions.npy
        metadata.json

train.py
    Dataset을 읽음
    collate_fn이 batch를 만듦
    policy.forward(...) 실행
    loss 계산
    checkpoint 저장

inference.py
    checkpoint 로드
    카메라 이미지와 state를 읽음
    policy.predict_action(...) 실행
    첫 번째 action을 로봇에 전달
```

## policy forward에서 전달되는 정보

ACT의 forward는 세 가지를 받습니다.

```python
actions = policy(images, language_ids, state)
```

| 입력 | 모양 | 의미 |
|---|---|---|
| `images` | `(B, num_cameras, 3, H, W)` | 여러 카메라가 본 현재 장면 |
| `language_ids` | `(B, text_len)` | 명령 문장을 숫자로 바꾼 토큰 |
| `state` | `(B, state_dim)` | 현재 로봇 관절 상태 |
| 반환값 | `(B, chunk_size, action_dim)` | 앞으로 `chunk_size`개 순간의 행동 |

pi0.5의 학습 loss는 네 가지 정보를 사용합니다.

```python
loss = policy.compute_flow_matching_loss(images, language_ids, ground_truth_actions)
```

| 정보 | 의미 |
|---|---|
| `images` | 장면을 보는 정보 |
| `language_ids` | 무엇을 하라는 명령인지 알려주는 정보 |
| `ground_truth_actions` | 사람이 보여준 정답 행동 |
| `x_t`, `t` | flow matching이 노이즈에서 행동으로 가는 중간 단계 |

더 자세한 정책 내부 흐름은 [policies/README.md](policies/README.md)에 정리되어 있습니다.

## 이번 검토에서 고친 주요 버그

| 영역 | 문제 | 수정 |
|---|---|---|
| 데이터 batch | `collate_fn`이 이미지 dict를 그대로 넘겨 `train.py`의 `.to(device)`에서 실패 | 모델 입력용 tensor로 변환 |
| 언어 입력 | 문자열이 policy로 직접 들어감 | ACT/pi0.5용 토큰 id 생성 |
| action chunk | 전체 episode action이 한 번에 들어감 | 현재 timestep부터 고정 길이 chunk로 자름 |
| state | 전체 state sequence가 들어갈 수 있음 | 현재 timestep state만 전달 |
| ACT loss mask | padding mask 의미가 어긋남 | `True = ignore` 기준으로 계산 |
| 학습 루프 | `accelerator=None`인데 `accelerator.is_main_process` 접근 | 단일 GPU/CPU에서도 동작하도록 guard 추가 |
| config | CLI로 만든 config를 버리고 기본 policy를 만들던 문제 | 실제 config로 policy 생성 |
| checkpoint | instance에서 `load_checkpoint`를 호출하고 반환값을 버림 | classmethod 반환값을 사용 |
| pi0.5 attention | forward마다 새 attention layer 생성 | 등록된 module로 변경 |
| pi0.5 sampling | flow sampler가 `forward`를 velocity 함수처럼 호출 | `_predict_velocity` 경로 추가 |
| robot collection | 매 프레임마다 `input()`이 걸려 녹화가 멈춤 | `Ctrl+C`로 종료하는 루프로 변경 |

## 자주 나는 문제

### `ModuleNotFoundError: No module named 'torch'`

의존성이 설치되지 않았습니다.

```bash
pip install -r requirements.txt
```

### `No ALOHA episodes found`

`--data-dir`이 episode 폴더를 포함한 디렉터리를 가리키는지 확인하세요.

```bash
python train.py --mode act --data-dir ./data/demo
```

### CUDA 메모리가 부족함

batch size를 줄이세요.

```bash
python train.py --mode act --batch-size 1
```

pi0.5는 기본 설정이 매우 큽니다. 먼저 ACT로 전체 흐름을 확인한 뒤 pi0.5를 줄인 설정으로 실험하는 것이 좋습니다.

## 파일 구조

```text
bh-vla/
├── collect_data.py
├── train.py
├── inference.py
├── policies/
│   ├── act.py
│   ├── pi05.py
│   ├── config.py
│   └── README.md
├── data/
│   ├── dataset.py
│   ├── transforms.py
│   └── robot_interface.py
├── utils.py
├── requirements.txt
└── README.md
```

## 참고 논문

- ACT: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- pi0.5: [pi0.5: A Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054)

## 라이선스와 사용 범위

이 저장소는 연구와 학습 목적의 코드입니다. 실제 로봇을 움직일 때는 반드시 낮은 속도, 안전한 공간, 비상 정지 절차를 먼저 준비하세요.
