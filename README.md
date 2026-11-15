# Fashion-MNIST 분류 모델 비교 프로젝트

Fashion-MNIST 의류 이미지 10종류를 분류하는 **완전연결 신경망(MLP)** 프로젝트입니다. 수업에서 배운 순전파, 오차역전파, 최적화 알고리즘, 가중치 초기화 등을 이용하고, **여러 모델을 비교해 검증 정확도가 높은 설정을 자동으로 선택**하도록 구현했습니다.

> **코드를 자세히 이해하고 싶다면:** [전체 코드 상세 설명서 — 데이터 처리·신경망 계산·Optimizer·12회 비교 실험·최종 모델 선택](docs/CODE_EXPLANATION.md)

## 프로젝트 핵심 기능

- **신경망 구조 비교 (4회)**: 은닉층 [50], [100], [100, 50]과 학습률 0.05, 0.10 조합 평가
- **Optimizer 자동 비교 (4회)**: SGD, Momentum, AdaGrad, Adam의 최고 검증 정확도, Epoch, 손실, 순위와 1·2위 차이 출력
- **Randomized Search (4회)**: 앞 단계에서 선택한 Optimizer로 학습률과 가중치 감소 계수 탐색
- **자동 모델 선정**: 최종 8 Epoch 중 검증 정확도가 가장 높은 시점의 모델 보관
- **결과 저장**: CSV 표, PNG 그래프, 최종 모델 PKL 파일 자동 생성

총 **12회 비교 실험 + 최종 학습 1회**를 진행합니다. 탐색은 앞 단계 결과를 다음 단계로 이어가는 **순차적 탐색**입니다. 모든 설정의 전체 조합을 전수 조사하는 방식은 아닙니다.

## 프로젝트 구조

~~~text
fashion-mnist-team/
├── fashion_mnist_team.py       # 전체 실험, 학습, 평가, 파일 저장
├── mnist_reader.py             # 공식 Fashion-MNIST 데이터 읽기
├── common/
│   ├── layers.py               # Affine, ReLU, SoftmaxWithLoss
│   ├── multi_layer_net.py      # MLP 모델, 순전파, 역전파, 정확도
│   └── optimizer.py            # SGD, Momentum, AdaGrad, Adam
├── docs/
│   └── CODE_EXPLANATION.md     # 함수별/실험별 상세 설명서
├── fashion-mnist/              # 처음 실행 시 데이터셋 자동 다운로드
└── Competition/                # 학습 후 결과가 저장되는 폴더
~~~

마지막 두 폴더는 실행 환경에 따라 없을 수 있으며, 프로그램을 실행하면서 생성하거나 다운로드합니다.

## 실행 방법

Python, Git이 설치되어 있어야 합니다.

~~~powershell
python -m pip install numpy matplotlib
python fashion_mnist_team.py
~~~

처음 실행할 때 [Fashion-MNIST 공식 GitHub](https://github.com/zalandoresearch/fashion-mnist)에서 데이터셋을 받습니다. 이미 `fashion-mnist` 폴더가 있지만 데이터 파일이 누락되었다면 경로와 파일 배치를 확인해야 합니다.

VS Code에서 기존 GitHub 저장소를 복제해 사용 중이라면 먼저 다음 명령을 실행할 수 있습니다.

~~~powershell
git pull origin main
~~~

단, 로컬에서 수정한 내용이 있다면 먼저 `git status`로 확인해 충돌이나 작업 손실을 방지하세요.

## 데이터와 모델 구성

| 항목 | 설정 |
|---|---|
| 데이터셋 | Fashion-MNIST, 28×28 흑백 이미지 |
| 분류 대상 | 10개 의류 클래스 |
| 입력 / 출력 | 784 / 10 |
| 은닉층 | 1~2개 구조 후보 비교 |
| 활성화 함수 | ReLU |
| 손실 함수 | Softmax + Cross-Entropy |
| 가중치 초기화 | He 초기화 |
| Batch Size | **100** |
| 후보 비교 | 각 **3 Epoch** |
| 최종 학습 | **8 Epoch** |
| 학습 데이터 | 54,000장 |
| 검증 데이터 | 6,000장 |
| 테스트 데이터 | 10,000장 |

후보 비교 때는 실행 시간을 줄이기 위해 학습 데이터 **6,000장**과 검증 데이터 **2,000장**만 사용합니다. 최종 학습은 학습 데이터 54,000장 전체로 진행하고 6,000장 검증 데이터로 최적 Epoch를 선택합니다. 테스트 데이터 10,000장은 **모델을 선택한 뒤 최종 평가**에 사용합니다.

## 실험 순서와 선택 기준

| 단계 | 비교 대상 | 횟수 | 선택 기준 |
|---|---|---:|---|
| 1. 수동 구조 탐색 | [50] / [100] / [100, 50]과 학습률 조합 | 4회 | 최고 검증 정확도를 기록한 **은닉층 구조** |
| 2. Optimizer 비교 | SGD / Momentum / AdaGrad / Adam | 4회 | 최고 검증 정확도의 **Optimizer** |
| 3. Randomized Search | 학습률 / Weight Decay | 4회 | 최고 검증 정확도의 **하이퍼파라미터** |
| 4. 최종 모델 학습 | 앞서 선택한 설정 | 1회 | 8 Epoch 중 가장 높은 검증 정확도의 모델 |
| 5. 최종 평가 | 별도 테스트 데이터 10,000장 | 1회 | 정확도 확인(선택 기준으로 사용하지 않음) |

Optimizer 비교는 **공통 학습률 0.01**, Weight Decay 0, 3 Epoch 조건으로 진행합니다. 따라서 여기서 1위를 한 Optimizer가 모든 조건에서 가장 우수하다는 뜻은 아닙니다.

Randomized Search에서는 다음 범위를 4번 탐색합니다.

- 학습률: 약 **0.01 ~ 0.1585**
- 가중치 감소 계수: **0.000001 ~ 0.001**

**코드 한 줄씩의 의미와 각 단계에서 실제 어떤 배열과 값이 사용되는지는 [상세 코드 설명서](docs/CODE_EXPLANATION.md)에 정리했습니다.**

## 생성되는 결과 파일

학습을 완료하면 `Competition/` 폴더에 아래 파일이 생성됩니다.

| 파일 | 의미 |
|---|---|
| `experiment_results.csv` | 구조 비교 4회와 Randomized Search 4회의 최고 검증 정확도 |
| `optimizer_results.csv` | Optimizer별 순위, 최고 검증 정확도, Epoch, 손실, 학습 정확도 |
| `optimizer_history.csv` | Optimizer별 Epoch의 손실/학습 정확도/검증 정확도 |
| `optimizer_accuracy.png` | 네 Optimizer의 검증 정확도 그래프 |
| `optimizer_loss.png` | 네 Optimizer의 손실 그래프 |
| `accuracy.png` | 최종 모델 학습·검증 정확도 그래프 |
| `loss.png` | 최종 학습 손실 그래프 |
| `network.pkl` | 선택한 최종 모델. 조 번호 설정 시 이름 변경 |

### 제출 모델 파일 이름

`fashion_mnist_team.py` 상단에서 조 번호를 지정합니다.

~~~python
TEAM_NUMBER = '5'  # 예: 5조
~~~

그러면 `Competition/network_Team5.pkl`로 저장됩니다. 비워 두면 `network.pkl`로 저장됩니다. Competition 평가 코드와의 호환 여부는 해당 평가 코드가 제공되면 별도 확인해야 합니다.

## 기존 실행에서 얻은 결과

이전 실행에서 다음 값을 기록했습니다.

| 평가 | 정확도 |
|---|---:|
| 최종 검증 정확도 | **86.25%** |
| 최종 테스트 정확도 | **84.55%** |

테스트 정확도 84.55%는 테스트 이미지 10,000장 중 **8,455장**을 맞혔다는 뜻입니다. 두 정확도의 차이는 **1.70%p**입니다.

이 값은 이전 실행에서 관찰된 결과이며, 최신 코드로 다시 실행한 경우의 선택된 Optimizer 및 최종 정확도는 **실제 터미널 출력과 CSV 기록**을 확인해야 합니다. 비교 과정에서 어느 후보가 몇 %였는지는 이 두 최종 수치만으로는 알 수 없습니다.

## 모델 비교를 해석할 때 주의할 점

- '가장 좋은 모델'은 **이번에 비교한 후보 중** 검증 정확도가 가장 높았던 모델을 뜻합니다.
- 12회 비교는 순차적 비교이며 모든 구조·Optimizer·하이퍼파라미터 조합을 시험한 것은 아닙니다.
- 학습 정확도와 학습 손실은 매 Epoch마다 **학습 데이터 중 2,000장 표본**으로 측정합니다.
- 최종 모델은 **8 Epoch 중 최고 검증 정확도 시점**의 모델입니다. 조기 종료(Early Stopping)는 아닙니다.
- 현재 모델은 CNN이 아니며, Dropout이나 Batch Normalization은 포함하지 않습니다.
- 자세한 구현상의 제약과 발표용 Q&A는 [상세 코드 설명서](docs/CODE_EXPLANATION.md)를 참고하세요.

## 참고 자료

- [Fashion-MNIST 공식 저장소](https://github.com/zalandoresearch/fashion-mnist)
- [공식 Fashion-MNIST 데이터 로더](https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py)
- [밑바닥부터 시작하는 딥러닝 참고 저장소](https://github.com/WegraLee/deep-learning-from-scratch)
- [Optimizer 참고 구현](https://github.com/WegraLee/deep-learning-from-scratch/blob/master/common/optimizer.py)
