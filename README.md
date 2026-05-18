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
| ACT | `policies/act.py` | 이미지를 보고 여러 개의 미래 행동을 한 번에 예측합니다. | LeRobot/ACT 논문 구조를 반영한 주력 구현 |
| pi0.5 | `policies/pi05.py` | 큰 VLM과 flow matching 방식의 행동 생성을 흉내 냅니다. | 연구용 스캐폴드 |

중요합니다. 이 코드는 논문을 완전히 재현한 공식 구현이 아닙니다. ACT는 LeRobot과 원 논문의 주요 학습 구조를 더 많이 반영했지만, 여전히 공식 LeRobot 패키지 전체를 대체하지는 않습니다. 특히 pi0.5는 PaliGemma 사전학습 가중치와 실제 tokenizer, 대규모 co-training 데이터가 없기 때문에 논문 성능을 기대하면 안 됩니다. 현재 목적은 구조를 이해하고 작은 데이터로 실험하는 것입니다.

## 논문 기준 검토

| 논문 | 코드에 들어간 핵심 아이디어 | 아직 빠진 부분 |
|---|---|---|
| ACT, arXiv:2304.13705 | action chunk 예측, multi-camera spatial image tokens, Transformer encoder/decoder, 현재 state 조건부 행동 예측, CVAE latent, KL loss, action queue, optional temporal ensemble | 최신 LeRobot Hub dataset/processor 전체 호환, 공식 ALOHA 평가/rollout 스택 |
| pi0.5, arXiv:2504.16054 | SigLIP 스타일 vision encoder, LLaMA 스타일 decoder, action expert, flow matching loss와 sampling 흐름 | 실제 PaliGemma 3B 가중치, 실제 tokenizer, 대규모 heterogeneous co-training, semantic subtask/object prediction, 논문 수준의 평가 코드 |

따라서 이 저장소를 읽을 때는 이렇게 보면 좋습니다.

- ACT는 이 저장소에서 가장 먼저 실험할 주력 imitation learning policy입니다.
- pi0.5는 "논문 구조를 따라가며 배우는 뼈대"에 가깝습니다.
- 논문과 같은 성능을 내려면 데이터 규모, robot calibration, LeRobot processor/Hub workflow, 평가 환경까지 맞춰야 합니다.

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

## SO-101 pick-and-place 운영 가이드

이번 주 실험 장비 구성을 기준으로 한 권장 흐름입니다.

```text
SO-101 leader arm + follower arm + top-view camera
    <-> MacBook
        데이터 수집
        checkpoint test
        실제 로봇 inference

MacBook
    <-> Ubuntu 24.04 training server
        rsync로 dataset/checkpoint 교환

Ubuntu 24.04 training server
        GPU 학습
```

### 준비 사항

양쪽 머신에서 같은 commit을 사용합니다.

```bash
git clone git@github.com-sphill9803:sphill9803/bh-vla.git
cd bh-vla
git pull origin main
```

MacBook은 로봇과 카메라 연결용입니다. Ubuntu server는 학습용입니다. 두 머신 모두 Python 환경을 만듭니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ubuntu server에서는 CUDA PyTorch가 잡히는지 먼저 확인합니다.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

MacBook에서는 카메라가 보이는지 먼저 확인합니다. 현재 collector는 기본적으로 OpenCV camera index `0`을 사용합니다.

```bash
python - <<'PY'
import cv2
cap = cv2.VideoCapture(0)
print("camera opened:", cap.isOpened())
cap.release()
PY
```

SO-101은 실제 inference 전에 반드시 낮은 속도, 손이 닿지 않는 작업 공간, 즉시 `Ctrl+C` 가능한 터미널 상태로 시작합니다.

### 이번 주 task 전략

처음 task는 하나만 고정합니다.

```text
task name: so101_pick_and_place_topview
instruction: pick up the red block and place it in the tray
camera: top-view 1개, 위치 고정
object: 색/크기/모양이 분명한 물체 1개
target: tray 또는 표시된 영역 1개
```

처음에는 다양성을 욕심내지 말고 성공 demonstration을 모으는 것이 좋습니다.

| 단계 | 목표 | 권장 데이터 |
|---|---|---|
| smoke | 코드 경로 확인 | synthetic 3 episodes |
| first robot check | 실제 수집/학습/inference 연결 확인 | 5 episodes |
| first overfit | 한 task를 외우는지 확인 | 20 episodes |
| first useful run | 위치 변화에 조금 버티는지 확인 | 50~100 episodes |

데이터 수집 전략은 다음처럼 가져갑니다.

- 같은 instruction 문장을 반복해서 사용합니다.
- top-view camera 위치와 조명을 고정합니다.
- 시작 위치는 조금씩 바꾸되, 처음 20 episodes는 너무 어렵게 만들지 않습니다.
- 실패한 episode는 학습 데이터에서 빼는 것이 좋습니다.
- pick, lift, move, place가 자연스럽게 이어지도록 leader arm을 부드럽게 움직입니다.
- episode 길이는 처음에는 5~15초 정도로 짧게 유지합니다.
- 물체가 카메라 밖으로 나가거나 손/팔이 과하게 가리는 episode는 버립니다.

### 1. MacBook에서 데이터 수집

로봇 없이 먼저 synthetic data로 저장 경로와 학습 코드가 도는지 확인합니다.

```bash
python collect_data.py \
  --mode act \
  --task-name so101_pick_and_place_topview_smoke \
  --num-episodes 3 \
  --skip-robot
```

실제 SO-101과 top-view camera로 수집합니다.

```bash
python collect_data.py \
  --mode act \
  --task-name so101_pick_and_place_topview \
  --num-episodes 50 \
  --image-size 224
```

각 episode 시작 때 같은 instruction을 입력합니다.

```text
pick up the red block and place it in the tray
```

수집 결과는 MacBook에 저장됩니다.

```text
data/so101_pick_and_place_topview/
├── dataset_info.json
├── episode_0000/
│   ├── images.npy
│   ├── states.npy
│   ├── actions.npy
│   └── metadata.json
└── episode_0001/
    └── ...
```

top-view camera 1개로 수집하면 `images.npy`는 보통 `(T, 224, 224, 3)`입니다. 현재 dataset loader는 단일 `images.npy`를 필요한 camera slot으로 복제해서 ACT 입력 `(B, num_cameras, 3, H, W)`를 만듭니다.

수집 직후 shape를 확인합니다.

```bash
python - <<'PY'
import json, os
import numpy as np

root = "data/so101_pick_and_place_topview/episode_0000"
print("images", np.load(os.path.join(root, "images.npy")).shape)
print("states", np.load(os.path.join(root, "states.npy")).shape)
print("actions", np.load(os.path.join(root, "actions.npy")).shape)
print(json.load(open(os.path.join(root, "metadata.json"))))
PY
```

문제가 없는 episode만 남깁니다. 불량 episode는 폴더를 지우거나 다른 곳으로 옮긴 뒤 학습합니다.

### 2. MacBook에서 Ubuntu server로 데이터 전송

예시는 Ubuntu server alias가 `trainbox`이고 repo 경로가 `~/workspace/bh-vla`인 경우입니다.

```bash
rsync -avP \
  data/so101_pick_and_place_topview/ \
  trainbox:~/workspace/bh-vla/data/so101_pick_and_place_topview/
```

IP로 직접 보낼 수도 있습니다.

```bash
rsync -avP \
  data/so101_pick_and_place_topview/ \
  user@192.168.0.20:~/workspace/bh-vla/data/so101_pick_and_place_topview/
```

반복 실험을 자주 할 경우 MacBook에 아래처럼 alias를 만들어두면 편합니다.

```bash
alias sync_so101_data='rsync -avP data/so101_pick_and_place_topview/ trainbox:~/workspace/bh-vla/data/so101_pick_and_place_topview/'
```

### 3. Ubuntu server에서 학습

먼저 작은 설정으로 smoke training을 합니다.

```bash
python train.py \
  --mode act \
  --data-dir ./data/so101_pick_and_place_topview \
  --data-format directory \
  --device cuda \
  --epochs 2 \
  --batch-size 2 \
  --chunk-size 16 \
  --n-action-steps 4
```

문제가 없으면 본학습을 실행합니다.

```bash
python train.py \
  --mode act \
  --data-dir ./data/so101_pick_and_place_topview \
  --data-format directory \
  --device cuda \
  --epochs 100 \
  --batch-size 16 \
  --chunk-size 100 \
  --n-action-steps 100
```

GPU 메모리가 부족하면 먼저 `--batch-size`를 줄입니다.

```bash
python train.py \
  --mode act \
  --data-dir ./data/so101_pick_and_place_topview \
  --data-format directory \
  --device cuda \
  --epochs 100 \
  --batch-size 4 \
  --chunk-size 100 \
  --n-action-steps 100
```

temporal ensemble을 실험하려면 `n_action_steps`를 1로 둡니다.

```bash
python train.py \
  --mode act \
  --data-dir ./data/so101_pick_and_place_topview \
  --data-format directory \
  --device cuda \
  --epochs 100 \
  --batch-size 16 \
  --chunk-size 100 \
  --n-action-steps 1 \
  --temporal-ensemble-coeff 0.01
```

학습 결과는 server에 저장됩니다.

```text
checkpoints/act_best.pt
checkpoints/act_last.pt
outputs/act_config.json
data/so101_pick_and_place_topview/dataset_stats.json
logs/train.log
```

`dataset_stats.json`은 학습 데이터의 action/state normalization 통계입니다. 실제 로봇 inference 품질과 안전에 중요하므로 checkpoint와 함께 MacBook으로 가져옵니다.

### 4. Ubuntu server에서 MacBook으로 결과 가져오기

MacBook에서 실행합니다.

```bash
mkdir -p checkpoints outputs data/so101_pick_and_place_topview

rsync -avP \
  trainbox:~/workspace/bh-vla/checkpoints/act_best.pt \
  ./checkpoints/act_best.pt

rsync -avP \
  trainbox:~/workspace/bh-vla/outputs/act_config.json \
  ./outputs/act_config.json

rsync -avP \
  trainbox:~/workspace/bh-vla/data/so101_pick_and_place_topview/dataset_stats.json \
  ./data/so101_pick_and_place_topview/dataset_stats.json
```

반복용 alias 예시입니다.

```bash
alias sync_so101_model='rsync -avP trainbox:~/workspace/bh-vla/checkpoints/act_best.pt ./checkpoints/act_best.pt && rsync -avP trainbox:~/workspace/bh-vla/data/so101_pick_and_place_topview/dataset_stats.json ./data/so101_pick_and_place_topview/dataset_stats.json'
```

### 5. MacBook에서 test inference

로봇을 움직이기 전에 checkpoint가 열리는지 확인합니다.

```bash
python inference.py \
  --mode act \
  --checkpoint ./checkpoints/act_best.pt \
  --test-mode \
  --device cpu \
  --language "pick up the red block and place it in the tray"
```

smoke training 때 `--chunk-size 16`으로 학습한 checkpoint라면 inference에도 맞춥니다.

```bash
python inference.py \
  --mode act \
  --checkpoint ./checkpoints/act_best.pt \
  --test-mode \
  --device cpu \
  --chunk-size 16 \
  --language "pick up the red block and place it in the tray"
```

### 6. MacBook에서 SO-101 inference

실제 로봇 실행은 처음에 매우 짧게 합니다.

```bash
python inference.py \
  --mode act \
  --checkpoint ./checkpoints/act_best.pt \
  --device cpu \
  --language "pick up the red block and place it in the tray"
```

처음 1~2초만 관찰하고 바로 `Ctrl+C`로 멈춥니다. action 방향, 그리퍼 움직임, 속도가 이상하면 학습 데이터와 action scale을 먼저 확인합니다.

현재 코드에서 특히 조심할 점은 inference pre/post-processing입니다. 학습에서는 `dataset_stats.json`을 사용해 action/state normalization을 적용하지만, `inference.py`는 아직 그 stats를 자동으로 읽어 state normalization과 action denormalization을 완전하게 처리하지 않습니다. 실제 로봇에서 안정적으로 쓰려면 다음 보강이 필요합니다.

```text
Mac inference:
    dataset_stats.json 로드
    current state normalize
    policy output action denormalize
    denormalized action을 SO-101에 send_action
```

이 보강 전에는 반드시 낮은 속도와 짧은 실행으로만 확인합니다.

### 권장 주간 루틴

첫날:

```text
synthetic 3 episodes
-> server smoke train
-> Mac test inference
```

둘째 날:

```text
real robot 5 episodes
-> server smoke train
-> Mac test inference
-> 실제 로봇 1~2초 저속 확인
```

셋째~넷째 날:

```text
성공 demonstration 20~50 episodes 수집
-> overfit 학습
-> 같은 초기 위치에서 성공률 확인
```

다섯째 날:

```text
초기 물체 위치를 조금씩 바꿔 50~100 episodes까지 확장
-> 본학습
-> 실패 케이스 기록
-> 실패 위치/각도 중심으로 추가 수집
```

성공률 기록은 간단히 표로 남깁니다.

| 날짜 | 데이터 | checkpoint | 테스트 조건 | 성공/시도 | 메모 |
|---|---:|---|---|---:|---|
| Day 2 | 5 eps | act_best.pt | 같은 위치 | 0/5 | action scale 확인 필요 |
| Day 3 | 25 eps | act_best.pt | 같은 위치 | 3/5 | place 단계 흔들림 |
| Day 5 | 80 eps | act_best.pt | 위치 변화 | 6/10 | 왼쪽 시작 위치 추가 필요 |

### task 확장 전략

pick-and-place 하나가 어느 정도 되기 전에는 task를 늘리지 않는 것이 좋습니다. 첫 task가 안정되면 다음 순서로 확장합니다.

1. 같은 물체, 같은 target, 시작 위치만 다양화
2. 같은 물체, target 위치 다양화
3. 색이 다른 같은 모양 물체 추가
4. instruction을 `"pick up the red block..."`, `"pick up the blue block..."`처럼 분리
5. 여러 물체가 동시에 보이는 장면으로 확장

각 task를 섞을 때는 instruction을 일관되게 적습니다. 같은 동작을 어떤 날은 `"pick red"`, 어떤 날은 `"grab red cube"`처럼 다르게 쓰면 작은 데이터에서는 오히려 학습이 어려워집니다.

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

ACT는 기본적으로 `chunk_size=100`, `n_action_steps=100`, CVAE/KL 학습을 사용합니다. 작은 GPU나 빠른 구조 확인이 필요하면 다음처럼 줄일 수 있습니다.

```bash
python train.py --mode act --data-dir ./data/demo --device cpu \
  --epochs 2 --batch-size 2 --chunk-size 16 --n-action-steps 4
```

temporal ensemble을 켜려면 매 step마다 policy를 호출해야 하므로 `--n-action-steps 1`을 같이 사용합니다.

```bash
python train.py --mode act --data-dir ./data/demo --device cuda \
  --n-action-steps 1 --temporal-ensemble-coeff 0.01
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
    ACT는 policy.forward(...)에서 action chunk와 VAE latent를 계산
    ACT loss는 masked L1 reconstruction + KL loss
    checkpoint 저장

inference.py
    checkpoint 로드
    카메라 이미지와 state를 읽음
    policy.predict_action(...) 실행
    action queue 또는 temporal ensemble으로 한 step action을 로봇에 전달
```

## policy forward에서 전달되는 정보

ACT의 forward는 세 가지를 받습니다.

```python
actions = policy(images, language_ids, state)
```

학습 중에는 정답 action chunk와 padding mask도 함께 전달해서 CVAE latent와 KL loss를 계산합니다.

```python
loss, metrics = policy.compute_loss(
    images,
    language_ids,
    state,
    ground_truth_actions,
    action_mask,
)
```

| 입력 | 모양 | 의미 |
|---|---|---|
| `images` | `(B, num_cameras, 3, H, W)` | 여러 카메라가 본 현재 장면 |
| `language_ids` | `(B, text_len)` | 명령 문장을 숫자로 바꾼 토큰 |
| `state` | `(B, state_dim)` | 현재 로봇 관절 상태 |
| `actions` | `(B, chunk_size, action_dim)` | 학습 때 쓰는 정답 미래 행동 |
| `action_mask` | `(B, chunk_size)` | `True`면 padding이라 loss에서 무시 |
| 반환값 | `(B, chunk_size, action_dim)` | 앞으로 `chunk_size`개 순간의 행동 |

ACT 내부에서는 여러 카메라 이미지를 ResNet spatial token으로 바꾸고, language token, state token, latent token과 함께 Transformer encoder에 넣습니다. decoder는 `chunk_size`개의 learnable action query를 사용해 미래 행동 chunk를 예측합니다.

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
| ACT 구조 | global pooled image feature와 단순 MSE loss만 사용 | spatial image tokens, CVAE latent, KL loss, masked L1 loss 추가 |
| ACT 추론 | 항상 chunk 첫 action만 사용 | `n_action_steps` action queue와 optional temporal ensemble 추가 |
| stats 적용 | 계산한 dataset stats가 dataset normalization에 일관되게 반영되지 않음 | `DatasetConfig`에 action/state stats를 주입 |

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
