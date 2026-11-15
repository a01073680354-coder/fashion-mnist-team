# Fashion-MNIST 코드 상세 설명서

이 문서는 현재 저장소의 소스 코드를 바탕으로 팀원들이 프로젝트의 학습 방식과 실험 결과 선택 과정을 이해할 수 있도록 작성했습니다. 기존 파일과 함수 이름을 그대로 사용합니다.

[메인 README](../README.md) · [메인 학습 코드](../fashion_mnist_team.py)

## 1. 프로젝트에서 구현한 내용

목표는 **28×28 흑백 의류 이미지를 10개 클래스로 구분하는 완전연결 신경망(MLP)** 을 만들고, 여러 후보를 비교하여 이번 실험에서 **검증 정확도가 높은 모델을 자동 선택**하는 것입니다.

전체 실행 순서:

1. 데이터셋 불러오기, 정규화, 학습/검증/테스트 분리
2. 은닉층 구성과 학습률 조합 **4회** 비교
3. SGD / Momentum / AdaGrad / Adam **4회** 비교
4. 선택된 구조·Optimizer를 이용한 Randomized Search **4회**
5. 가장 좋은 설정으로 전체 학습 데이터 **54,000장, 8 Epoch** 학습
6. 검증 정확도가 가장 좋은 Epoch의 모델을 선택
7. 별도 테스트 데이터 **10,000장**으로 최종 평가
8. 결과 CSV, 그래프 PNG, 모델 PKL 저장

총 **12회의 후보 비교 실험 + 최종 학습 1회**입니다. 이 숫자는 모두 같은 조건으로 비교한 후보 12개가 아니라, **앞 단계 결과를 다음 단계에 적용하는 순차적인 실험 12회**를 의미합니다.

## 2. 파일별 역할

| 파일 | 역할 |
|---|---|
| [fashion_mnist_team.py](../fashion_mnist_team.py) | 데이터 처리, 12회 비교, 최종 학습, 성능 출력 및 결과 저장 |
| [mnist_reader.py](../mnist_reader.py) | 공식 Fashion-MNIST의 gzip/IDX 이미지와 레이블 읽기 |
| [common/multi_layer_net.py](../common/multi_layer_net.py) | 신경망 구성, 예측, 정확도, 손실, 역전파 |
| [common/layers.py](../common/layers.py) | Affine, ReLU, SoftmaxWithLoss 기본 계층 |
| [common/optimizer.py](../common/optimizer.py) | SGD, Momentum, AdaGrad, Adam의 가중치 업데이트 |
| Competition/ | 학습 후 생성되는 결과 CSV, 이미지, 모델 파일 |
| fashion-mnist/ | 공식 GitHub 저장소에서 내려받은 데이터와 유틸리티 |

`main()` 이 프로젝트의 전체 실행 순서를 제어하며, `train_network()` 는 주어진 설정에 따라 신경망을 학습하고 검증하는 공통 함수입니다.

## 3. 데이터 로드: load_fashion_mnist()

### 저장 경로

`BASE_DIR = Path(__file__).resolve().parent` 로 현재 파이썬 파일 위치를 기준으로 경로를 지정합니다.

- GitHub 저장소를 복제하는 위치: `BASE_DIR / 'fashion-mnist'`
- 원본 데이터 위치: `BASE_DIR / 'fashion-mnist' / 'data' / 'fashion'`

`train-images-idx3-ubyte.gz` 파일이 없고 fashion-mnist 폴더도 없다면 다음 명령에 해당하는 동작을 수행합니다.

```powershell
git clone --depth 1 https://github.com/zalandoresearch/fashion-mnist.git fashion-mnist
```

**주의:** 폴더가 이미 있지만 필요한 데이터 파일이 없다면 자동으로 다시 복제하지 않고 `FileNotFoundError`를 발생시킵니다.

### 데이터 읽기

```python
x_train, t_train = load_mnist(str(data_folder), kind='train')
x_test, t_test = load_mnist(str(data_folder), kind='t10k')
```

`mnist_reader.py`는 gzip 압축을 풀어 IDX 파일의 헤더를 제외하고 이미지·레이블을 NumPy 배열로 읽습니다. 원본 학습 데이터 60,000장, 테스트 데이터 10,000장을 가져옵니다.

- 이미지 원본 크기: 28×28
- 한 장을 펼친 입력 벡터: 784개 값
- 정답 레이블: 정수 0~9
- 입력 배열 형태: `(이미지 수, 784)`

### 정규화

```python
x_train.astype(np.float32) / 255.0
```

픽셀의 0~255 범위를 0~1로 변경합니다. 입력 크기를 정리하여 신경망 학습에 사용합니다.

## 4. 데이터 분리: main()

```python
rng = np.random.RandomState(2026)
ids = rng.permutation(len(t))
val_ids, train_ids = ids[:6000], ids[6000:]
```

원본 학습 데이터 60,000장을 무작위로 섞고 다음과 같이 분리합니다.

| 데이터 | 개수 | 용도 |
|---|---:|---|
| 학습 | 54,000장 | 가중치와 편향을 업데이트 |
| 검증 | 6,000장 | 모델/하이퍼파라미터/최적 Epoch 선정 |
| 테스트 | 10,000장 | 최종 모델 선정 후 성능 평가 |

비교 시간을 줄이기 위해 12회의 비교 실험에는 학습 데이터 중 6,000장, 검증 데이터 중 2,000장만 사용합니다.

```python
small_x, small_t = x_train[:6000], t_train[:6000]
small_val_x, small_val_t = x_val[:2000], t_val[:2000]
```

- **비교 실험 점수**: 검증 이미지 **2,000장** 기준
- **최종 학습의 검증 점수**: 검증 이미지 **6,000장** 기준
- **최종 테스트 점수**: 테스트 이미지 **10,000장** 기준

따라서 비교 실험의 검증 정확도와 최종 학습의 검증 정확도는 **평가 데이터 수가 다르다**는 점을 결과표에서 구분해야 합니다.

## 5. MultiLayerNet: 신경망 구성

모델 생성 코드:

```python
net = MultiLayerNet(
    784, hidden, 10,
    weight_init_std='he',
    weight_decay_lambda=decay
)
```

- 입력층: 784차원
- 은닉층: 리스트 `hidden`에 따라 결정
- 출력층: 10차원
- 은닉층 활성화: ReLU
- 출력 손실: SoftmaxWithLoss
- 가중치 초기화: He 초기화
- 정규화: Weight Decay(L2)

예를 들어 `hidden = [100, 50]`이라면 다음과 같습니다.

```text
입력(784) → Affine → ReLU(100) → Affine → ReLU(50) → Affine → 출력(10)
```

이 구조의 파라미터 배열 크기:

| 파라미터 | 크기 | 의미 |
|---|---|---|
| W1 | (784, 100) | 입력 → 첫 번째 은닉층 |
| b1 | (100,) | 첫 번째 은닉층 편향 |
| W2 | (100, 50) | 첫 번째 → 두 번째 은닉층 |
| b2 | (50,) | 두 번째 은닉층 편향 |
| W3 | (50, 10) | 두 번째 은닉층 → 출력층 |
| b3 | (10,) | 출력층 편향 |

### He 초기화

```python
scale = np.sqrt(2.0 / all_sizes[i - 1])
self.params['W' + str(i)] = scale * np.random.randn(...)
```

이전 계층의 뉴런 수를 이용해 초기 가중치의 크기를 조절하는 방식입니다.

**구현 시 주의:** 현재 `MultiLayerNet` 클래스는 `activation`과 `weight_init_std`를 생성자 인자로 받지만 내부에서는 실제로 **ReLU와 He 초기화로 고정**되어 있습니다. 이 인자만 바꾼다고 다른 활성화 함수나 초기화가 적용되지는 않습니다.

## 6. layers.py: 계층의 순전파와 역전파

### Affine 계층

완전연결 계산을 담당합니다.

```python
out = np.dot(self.x, self.W) + self.b
```

역전파:

```python
dx = np.dot(dout, self.W.T)
self.dW = np.dot(self.x.T, dout)
self.db = np.sum(dout, axis=0)
```

- `dx`: 이전 계층에 전달할 기울기
- `dW`: 가중치에 대한 기울기
- `db`: 편향에 대한 기울기

### ReLU 계층

```python
self.mask = (x <= 0)
out[self.mask] = 0
```

0 이하 입력은 0으로, 양수는 그대로 출력합니다. 역전파에서도 마스크를 이용해 해당 위치의 기울기를 0으로 만듭니다.

### SoftmaxWithLoss 계층

- `softmax(x)`: 출력 점수를 클래스별 확률로 변환
- `cross_entropy_error(y, t)`: 정답에 대한 오차 계산
- `backward()`: 오차 기울기를 계산하여 이전 계층으로 전달

`MultiLayerNet.predict()`는 softmax 전의 출력 점수를 반환합니다. 분류 결과는 `argmax`로 선택하며, softmax 적용 전후에 최대값의 인덱스는 동일합니다.

### MultiLayerNet의 주요 메서드

| 메서드 | 용도 |
|---|---|
| `predict(x)` | 각 클래스의 출력 점수 계산 |
| `loss(x, t)` | 교차 엔트로피 손실 및 가중치 감소 벌점 계산 |
| `accuracy(x, t)` | 실제 레이블과 예측 레이블의 일치 비율 |
| `gradient(x, t)` | 역전파로 W, b의 기울기 계산 |

### Weight Decay

```python
weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)
```

큰 가중치에 벌점을 주는 L2 정규화 방식입니다. `gradient()`에서도 가중치 기울기에 `weight_decay_lambda * W`를 더합니다.

## 7. train_network(): 학습의 핵심 함수

```python
def train_network(hidden, lr, decay, x_train, t_train, x_val, t_val,
                  epochs=3, optimizer_name='SGD'):
```

| 인수 | 역할 |
|---|---|
| `hidden` | 예: `[100, 50]`처럼 은닉층 수·크기 지정 |
| `lr` | 학습률 |
| `decay` | 가중치 감소 계수 |
| `x_train, t_train` | 학습용 이미지와 정답 |
| `x_val, t_val` | 검증용 이미지와 정답 |
| `epochs` | 전체 학습 데이터 반복 횟수 |
| `optimizer_name` | SGD / Momentum / AdaGrad / Adam 중 하나 |

### 7-1. 모델과 Optimizer 생성

```python
np.random.seed(7)
batch_size = 100
optimizer_classes = {
    'SGD': SGD, 'Momentum': Momentum,
    'AdaGrad': AdaGrad, 'Adam': Adam
}
optimizer = optimizer_classes[optimizer_name](lr=lr)
```

같은 구조의 모델 비교에서 초기 가중치와 배치 순서의 랜덤 조건을 동일하게 맞춥니다.

### 7-2. Mini-Batch 학습

```python
for epoch in range(epochs):
    order = np.random.permutation(len(x_train))
    for i in range(0, len(x_train), batch_size):
        ids = order[i:i + batch_size]
        grads = net.gradient(x_train[ids], t_train[ids])
        optimizer.update(net.params, grads)
```

실행 과정:

1. 매 Epoch 학습 이미지 순서를 무작위로 섞음
2. 이미지 100장씩 Mini-Batch 생성
3. 오차역전파로 각 파라미터의 기울기 계산
4. 선택한 Optimizer가 가중치와 편향 업데이트
5. 모든 학습 이미지를 사용하면 1 Epoch 종료

Batch Size는 **100**입니다.

- 비교 실험: 6,000 ÷ 100 = Epoch당 **60회** 업데이트
- 최종 학습: 54,000 ÷ 100 = Epoch당 **540회** 업데이트
- 최종 학습 8 Epoch: **4,320회** 업데이트

### 7-3. Epoch별 측정

```python
train_acc = net.accuracy(x_train[:2000], t_train[:2000])
val_acc = net.accuracy(x_val, t_val)
loss = net.loss(x_train[:2000], t_train[:2000])
```

- 학습 정확도와 손실: 속도를 위해 **학습 이미지 앞 2,000장**만 사용
- 검증 정확도: 전달된 검증 데이터 전체(비교 시 2,000장, 최종 시 6,000장)

따라서 출력되는 train 정확도는 전체 54,000장의 정확도가 아니라 **일부 이미지에 대한 측정치**입니다.

### 7-4. 최고 정확도 모델 보관

```python
if val_acc > best_acc:
    best_acc, best_net = val_acc, copy.deepcopy(net)
```

매 Epoch 검증 정확도를 비교해 가장 높은 점수일 때의 네트워크를 저장합니다. 즉 **마지막 Epoch 모델을 무조건 선택하지 않습니다.**

단, 실제 학습은 설정된 Epoch까지 끝까지 진행합니다. 이것은 **최고 성능 체크포인트 선택**이며 학습을 일찍 종료하는 Early Stopping은 아닙니다.

함수 반환값:

```python
return best_acc, best_net, losses, train_accs, val_accs
```

`best_net`은 최고 검증 정확도 시점 모델이고, `losses`와 정확도 리스트는 **전체 Epoch의 기록**입니다.

## 8. Optimizer 4가지 설명

구현 파일: [common/optimizer.py](../common/optimizer.py)

### SGD (Stochastic Gradient Descent)

```python
params[key] -= self.lr * grads[key]
```

현재 기울기에 학습률을 곱해 해당 방향으로 가중치를 수정합니다. 가장 기본적인 비교 기준입니다.

### Momentum

```python
self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
params[key] += self.v[key]
```

이전 가중치 이동 방향을 반영해 관성을 추가합니다. 기본 Momentum 값은 **0.9**입니다.

### AdaGrad

```python
self.h[key] += grads[key] ** 2
params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
```

기울기 제곱의 누적합을 이용해 파라미터별 업데이트 크기를 조절합니다. 누적합이 계속 커지면 학습 후반에 업데이트가 작아질 수 있습니다.

### Adam

```python
self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])
params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
```

기울기와 기울기 제곱의 이동 평균을 모두 사용합니다. 기본 `beta1=0.9`, `beta2=0.999`이며 반복 횟수를 반영하는 보정 학습률 `lr_t`가 적용됩니다.

> 중요한 구분: 이 프로젝트의 Optimizer 비교는 **모두 공통 학습률 0.01**을 사용합니다. 따라서 네 알고리즘이 각자 최적 학습률로 경쟁한 것은 아니며, **이번 실험 조건에서의 성능 비교**입니다.

## 9. 1단계: 구조·학습률 수동 비교 (4회)

```python
manual = [
    ([50], 0.05),
    ([100], 0.05),
    ([100, 50], 0.05),
    ([100, 50], 0.10)
]
```

| 후보 | 은닉층 구성 | 은닉층 개수 | 학습률 | Optimizer |
|---|---|---:|---:|---|
| A | [50] | 1 | 0.05 | SGD |
| B | [100] | 1 | 0.05 | SGD |
| C | [100, 50] | 2 | 0.05 | SGD |
| D | [100, 50] | 2 | 0.10 | SGD |

각 후보를 학습 6,000장, 검증 2,000장, 3 Epoch로 평가합니다.

```python
best_manual = max(results, key=lambda row: row[4])
hidden = parse_hidden(best_manual[1])
```

`results` 각 행의 5번째 요소(`row[4]`)가 최고 검증 정확도입니다. 정확도가 가장 높은 후보에서 **은닉층 구성만** 다음 단계로 전달합니다.

**주의:** 이 단계의 학습률은 이후 그대로 유지되지 않습니다. Optimizer 비교는 학습률을 0.01로 고정하고, Randomized Search에서 학습률을 다시 탐색합니다.

## 10. 2단계: Optimizer 자동 비교 (4회)

```python
optimizer_names = ['SGD', 'Momentum', 'AdaGrad', 'Adam']
```

각 알고리즘을 동일한 은닉층, 동일한 초기화 난수 시드, 동일한 데이터, **학습률 0.01**, 3 Epoch로 학습합니다.

알고리즘별로 기록하는 내용:

- 최고 검증 정확도
- 최고 검증 정확도가 나온 Epoch
- 해당 Epoch의 학습 정확도(2,000장 기준)
- 해당 Epoch의 손실
- 마지막 Epoch의 검증 정확도
- Epoch별 전체 기록

```python
best_epoch = int(np.argmax(val_accs)) + 1
optimizer_results.append([
    name, float(score), best_epoch,
    float(train_accs[best_epoch - 1]),
    float(losses[best_epoch - 1]),
    float(val_accs[-1])
])
```

### 정확도 순위와 우승 알고리즘 결정

```python
optimizer_ranking = sorted(
    optimizer_results, key=lambda row: row[1], reverse=True
)
best_optimizer = optimizer_ranking[0][0]
```

최고 검증 정확도의 내림차순으로 정렬하고 1위를 선택합니다.

프로그램은 **1~4위 순위표**, 1위 알고리즘과 검증 정확도, 2위와의 **정확도 차이(%p)** 를 출력합니다. 점수가 같으면 원래 목록에서 먼저 있던 알고리즘이 앞에 옵니다.

실제 순위와 정확도는 실행할 때 계산되므로, 어떤 알고리즘이 1위였는지는 결과 파일이나 터미널 로그에서 확인해야 합니다.

## 11. 3단계: Randomized Search (4회)

1단계에서 선택한 은닉층, 2단계에서 선택한 Optimizer로 학습률과 가중치 감소 계수를 탐색합니다.

```python
random_rng = np.random.RandomState(42)
for _ in range(4):
    lr = float(10 ** random_rng.uniform(-2.0, -0.8))
    decay = float(10 ** random_rng.uniform(-6.0, -3.0))
```

| 하이퍼파라미터 | 범위 |
|---|---|
| 학습률 | 0.01 ~ 약 0.1585 |
| 가중치 감소 계수 | 0.000001 ~ 0.001 |
| 실험 횟수 | 4회 |
| Epoch | 3 |

각 후보를 검증 정확도로 평가하고 다음 코드로 1위를 선택합니다.

```python
best = max(results[len(manual):], key=lambda row: row[4])
final_hidden, final_lr, final_decay = (
    parse_hidden(best[1]), best[2], best[3]
)
```

`results` 앞부분에는 수동 비교 4회의 결과가 있으므로 `results[len(manual):]`를 사용해 **Randomized Search 결과 4개만 추출**합니다.

4개 후보 중 1위를 선택하는 방식이므로, 전체 가능한 학습률의 전역 최적해를 찾았다는 의미는 아닙니다.

## 12. 4단계: 최종 모델 재학습과 선정

```python
acc, network, losses, train_accs, val_accs = train_network(
    final_hidden, final_lr, final_decay,
    x_train, t_train, x_val, t_val,
    epochs=8, optimizer_name=best_optimizer
)
```

- 은닉층: 1단계에서 선택
- Optimizer: 2단계에서 선택
- 학습률과 Weight Decay: 3단계에서 선택
- 최종 학습 데이터: **54,000장**
- 최종 검증 데이터: **6,000장**
- 최종 학습: **8 Epoch**
- 최고 검증 정확도를 기록한 Epoch의 `best_net` 사용

### 테스트 정확도

```python
test_acc = network.accuracy(x_test, t_test)
```

최종 선택이 끝난 뒤 별도 **10,000장 테스트 데이터**를 사용하여 성능을 계산합니다. 테스트 정확도는 학습 도중 후보를 선택할 때 사용하지 않습니다.

### 모델 저장

```python
filename = f'network_Team{TEAM_NUMBER}.pkl' if TEAM_NUMBER else 'network.pkl'
with (out / filename).open('wb') as f:
    pickle.dump(network, f)
```

- `TEAM_NUMBER = ''`: `Competition/network.pkl` 저장
- `TEAM_NUMBER = '5'`: `Competition/network_Team5.pkl` 저장

교수님 측 Competition 코드에서 사용할 수 있는지 여부는 **해당 평가 코드를 이용해 별도로 확인**해야 합니다.

## 13. 결과 파일과 해석

학습이 끝나면 `Competition` 폴더가 생성됩니다.

```text
Competition/
├── experiment_results.csv
├── optimizer_results.csv
├── optimizer_history.csv
├── optimizer_accuracy.png
├── optimizer_loss.png
├── loss.png
├── accuracy.png
└── network.pkl           # TEAM_NUMBER를 지정했다면 network_TeamN.pkl
```

| 파일 | 저장 내용 | 용도 |
|---|---|---|
| `experiment_results.csv` | 구조 비교 4회 + Randomized Search 4회 결과 | 후보 설정 비교 |
| `optimizer_results.csv` | Optimizer 순위, 최고 검증 정확도, 최고 Epoch, 당시 학습 정확도·손실 등 | 알고리즘 선정 근거 |
| `optimizer_history.csv` | 4개 Optimizer의 Epoch별 손실·학습/검증 정확도 | 변화 과정 분석 |
| `optimizer_accuracy.png` | 네 Optimizer의 검증 정확도 변화 | 발표용 그래프 |
| `optimizer_loss.png` | 네 Optimizer의 손실 변화 | 학습 특성 분석 |
| `loss.png` | 최종 8 Epoch의 손실 | 최종 학습 경과 |
| `accuracy.png` | 최종 학습·검증 정확도 변화 | 일반화/과적합 확인 |
| `network.pkl` 또는 `network_TeamN.pkl` | 최고 검증 정확도 Epoch 모델 객체 | 최종 평가 및 제출 |

CSV에서 정확도는 **0~1 소수**로 기록합니다. 예를 들어 0.8625는 **86.25%**입니다.

### 과거 실행 기록 해석

한 차례 실행에서 기록된 결과는 **최종 검증 정확도 86.25%, 테스트 정확도 84.55%**였습니다.

- 검증 데이터 6,000장의 약 86.25% 정확도: 5,175장 정답
- 테스트 데이터 10,000장의 84.55% 정확도: 8,455장 정답
- 차이: **1.70%p**

이는 한 실행에서 측정된 값이며, 최신 실행의 선택된 Optimizer 및 정확도는 **실제 출력 결과로 다시 확인**하는 것이 좋습니다.

## 14. 재현성과 실험 해석 시 유의할 점

### 랜덤 시드

- `np.random.RandomState(2026)`: 학습/검증 데이터 분리
- `np.random.seed(7)`: 모델 가중치 초기화 및 배치 순서
- `np.random.RandomState(42)`: Randomized Search 후보 생성

실험 조건을 일정하게 맞추기 위한 장치입니다. 다만 실행 환경의 차이로 수치가 약간 달라질 수 있습니다.

### 정확한 장점과 한계

**장점**

1. 여러 구조·Optimizer·학습 조건을 한 번의 실행으로 비교
2. 검증 정확도에 근거한 자동 순위 산출
3. 선택 결과를 다음 실험으로 연결
4. 최종 8 Epoch 중 가장 정확한 시점의 모델 보관
5. CSV·PNG·PKL로 결과 저장, 발표 근거 확보

**한계**

1. 은닉층 구조는 미리 정한 **4개 조합**만 확인
2. Optimizer는 모두 **학습률 0.01, 3 Epoch**로 비교: 각 알고리즘의 최적 설정을 찾은 것은 아님
3. Randomized Search는 **4회**만 탐색
4. 12회 비교는 **순차적 탐색**이지 모든 조합의 전수 비교가 아님
5. 다중 랜덤 시드 반복 실험, Early Stopping, CNN, Dropout, Batch Normalization 등은 포함되지 않음
6. 최종 정확도 그래프의 학습 정확도와 손실은 학습 데이터 **2,000장 표본** 기준
7. 모델로 선택된 Epoch와 그래프의 마지막 Epoch가 다를 수 있음

발표할 때 '모든 경우의 최적 모델'이 아니라 **'이번에 비교한 후보 중 검증 정확도가 가장 높은 모델'**이라고 설명해야 정확합니다.

### 실행 중 나타나는 특징

- `common/layers.py`의 `Relu` 클래스 본문에는 학습용 테스트 배열을 출력하는 `print` 코드가 남아 있어 import할 때 NumPy 배열이 출력될 수 있습니다. 이는 실험 정확도 출력과 별개입니다.
- 그래프는 최종 학습 후 `plt.show()`로 표시됩니다. 화면 창이 열리지 않아도 PNG 파일이 저장되어 있다면 파일 탐색기에서 확인할 수 있습니다.
- `pickle.load()`로 읽는 모델 파일은 신뢰할 수 있는 것만 사용해야 하며, 같은 클래스 모듈을 import할 수 있는 환경이 필요합니다.

## 15. 발표할 때 설명하는 순서

**문제:** 은닉층의 개수, 학습률, 최적화 방법에 따라 신경망의 분류 정확도가 달라집니다.

**방법:** 구조·학습률 4회, SGD/Momentum/AdaGrad/Adam 4회, Randomized Search 4회로 총 **12회 비교 실험**을 자동 수행하도록 구현했습니다.

**선정:** 각 단계에서 **최고 검증 정확도**를 비교해 다음 단계의 후보를 결정했습니다.

**최종 학습:** 선정된 설정을 학습 데이터 54,000장으로 8 Epoch 학습하고 가장 좋은 Epoch의 모델을 저장했습니다.

**평가:** 테스트 데이터 10,000장은 최종 평가에만 사용했습니다. 실험 결과는 순위표, CSV, 그래프에 남겨 선정 이유를 확인할 수 있습니다.

> **핵심 메시지:** "하나의 모델을 임의로 만드는 것이 아니라, 여러 후보의 검증 정확도를 비교하고 결과를 자동으로 정리하여 최종 모델을 선정하도록 구성했습니다."

## 참고 링크

- [Fashion-MNIST 공식 저장소](https://github.com/zalandoresearch/fashion-mnist)
- [공식 데이터 로더](https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py)
- [밑바닥부터 시작하는 딥러닝 참고 코드](https://github.com/WegraLee/deep-learning-from-scratch)
- [Optimizer 구현 참고](https://github.com/WegraLee/deep-learning-from-scratch/blob/master/common/optimizer.py)
