# Fashion-MNIST 팀프로젝트: 4, 5, 6장 실습 내용으로 만든 신경망
# 데이터: https://github.com/zalandoresearch/fashion-mnist
# 신경망: https://github.com/WegraLee/deep-learning-from-scratch

import csv
import copy
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common.multi_layer_net import MultiLayerNet


# 조 번호는 나중에 직접 작성 (예: '6')
TEAM_NUMBER = ''
BASE_DIR = Path(__file__).resolve().parent


def load_fashion_mnist():
    """GitHub의 Fashion-MNIST를 받아 공식 mnist_reader.py로 읽는다."""
    github_folder = BASE_DIR / 'fashion-mnist'
    data_folder = github_folder / 'data' / 'fashion'

    if not (data_folder / 'train-images-idx3-ubyte.gz').exists():
        if github_folder.exists():
            raise FileNotFoundError('fashion-mnist 폴더에 데이터가 없습니다. 폴더를 확인해 주세요.')
        print('GitHub에서 Fashion-MNIST를 받습니다.')
        try:
            subprocess.run([
                'git', 'clone', '--depth', '1',
                'https://github.com/zalandoresearch/fashion-mnist.git',
                str(github_folder)
            ], check=True)
        except FileNotFoundError as exc:
            raise RuntimeError('Git이 설치되어 있지 않습니다. Git 설치 후 다시 실행하세요.') from exc

    # 공식 GitHub 저장소의 로더 사용: 기존 dataset/mnist.py와 다름!
    sys.path.insert(0, str(github_folder / 'utils'))
    import mnist_reader

    x_train, t_train = mnist_reader.load_mnist(str(data_folder), kind='train')
    x_test, t_test = mnist_reader.load_mnist(str(data_folder), kind='t10k')
    return x_train.astype(np.float32) / 255.0, t_train, x_test.astype(np.float32) / 255.0, t_test


def train_network(hidden, lr, decay, x_train, t_train, x_val, t_val, epochs=3):
    """5장 오차역전파와 6장 SGD를 사용해 학습한다."""
    np.random.seed(7)  # 모델 비교 조건을 같게 설정
    net = MultiLayerNet(784, hidden, 10, weight_init_std='he', weight_decay_lambda=decay)
    losses, train_accs, val_accs = [], [], []
    best_acc, best_net = -1, None
    batch_size = 100

    for epoch in range(epochs):
        order = np.random.permutation(len(x_train))
        for i in range(0, len(x_train), batch_size):
            ids = order[i:i + batch_size]
            grads = net.gradient(x_train[ids], t_train[ids])
            for key in net.params:
                net.params[key] -= lr * grads[key]

        # 데이터가 많을 때도 출력이 느리지 않도록 훈련 정확도는 앞 2,000장으로 확인
        train_acc = net.accuracy(x_train[:2000], t_train[:2000])
        val_acc = net.accuracy(x_val, t_val)
        loss = net.loss(x_train[:2000], t_train[:2000])
        losses.append(float(loss))
        train_accs.append(float(train_acc))
        val_accs.append(float(val_acc))
        print(f'  epoch {epoch + 1}: loss={loss:.4f}, train={train_acc:.3f}, val={val_acc:.3f}')
        if val_acc > best_acc:
            best_acc, best_net = val_acc, copy.deepcopy(net)

    return best_acc, best_net, losses, train_accs, val_accs


def parse_hidden(text):
    # CSV에 적은 [100, 50] 형태의 은닉층 구조를 리스트로 복원
    return [int(value.strip()) for value in text.strip('[]').split(',')]


def main():
    x, t, x_test, t_test = load_fashion_mnist()

    # 전체 훈련 60,000장을 학습/검증으로 분리. 테스트 10,000장은 마지막에만 사용.
    rng = np.random.RandomState(2026)
    ids = rng.permutation(len(t))
    val_ids, train_ids = ids[:6000], ids[6000:]
    x_train, t_train = x[train_ids], t[train_ids]
    x_val, t_val = x[val_ids], t[val_ids]
    del x, t
    print('학습:', x_train.shape, '검증:', x_val.shape, '테스트:', x_test.shape)

    # 실험은 시간을 아끼기 위해 학습 6,000장/검증 2,000장으로 진행
    small_x, small_t = x_train[:6000], t_train[:6000]
    small_val_x, small_val_t = x_val[:2000], t_val[:2000]
    results = []

    print('\n[1] 수동 비교: 은닉층 구조와 학습률')
    manual = [([50], 0.05), ([100], 0.05), ([100, 50], 0.05), ([100, 50], 0.10)]
    for hidden, lr in manual:
        acc, _, _, _, _ = train_network(hidden, lr, 0, small_x, small_t, small_val_x, small_val_t)
        results.append(['수동', str(hidden), lr, 0, float(acc)])
        print(' 결과:', hidden, lr, '검증 정확도', round(float(acc), 4))

    # 수동 탐색에서 검증 정확도가 가장 높은 구조를 선택
    best_manual = max(results, key=lambda row: row[4])
    hidden = parse_hidden(best_manual[1])

    print('\n[2] Randomized Search: 학습률과 가중치 감소를 4번 무작위 탐색')
    random_rng = np.random.RandomState(42)
    for _ in range(4):
        lr = float(10 ** random_rng.uniform(-2.0, -0.8))
        decay = float(10 ** random_rng.uniform(-6.0, -3.0))
        acc, _, _, _, _ = train_network(hidden, lr, decay, small_x, small_t, small_val_x, small_val_t)
        results.append(['랜덤', str(hidden), lr, decay, float(acc)])
        print(' 결과:', round(lr, 5), round(decay, 6), '검증 정확도', round(float(acc), 4))

    out = BASE_DIR / 'Competition'
    out.mkdir(exist_ok=True)
    with (out / 'experiment_results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['방식', '은닉층', '학습률', '가중치감소', '검증정확도'])
        writer.writerows(results)

    best = max(results, key=lambda row: row[4])
    final_hidden, final_lr, final_decay = parse_hidden(best[1]), best[2], best[3]
    print('\n[3] 최종 모델 학습:', best[:4])
    acc, network, losses, train_accs, val_accs = train_network(
        final_hidden, final_lr, final_decay,
        x_train, t_train, x_val, t_val, epochs=8
    )

    plt.figure()
    plt.plot(losses)
    plt.title('Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(out / 'loss.png')
    plt.close()

    plt.figure()
    plt.plot(train_accs, label='train')
    plt.plot(val_accs, label='validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(out / 'accuracy.png')
    plt.close()

    # 테스트 세트는 모든 모델 선택이 끝난 뒤에 단 한 번 평가
    test_acc = network.accuracy(x_test, t_test)
    print('\n최종 검증 정확도:', round(float(acc), 4))
    print('최종 테스트 정확도:', round(float(test_acc), 4))

    filename = f'network_Team{TEAM_NUMBER}.pkl' if TEAM_NUMBER else 'network.pkl'
    with (out / filename).open('wb') as f:
        pickle.dump(network, f)
    print('저장 완료:', out / filename)


if __name__ == '__main__':
    main()
