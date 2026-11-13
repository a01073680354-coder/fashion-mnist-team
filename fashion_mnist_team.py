# Fashion-MNIST 팀플 코드
# 데이터 출처: https://github.com/zalandoresearch/fashion-mnist
# 참고 코드: https://github.com/WegraLee/deep-learning-from-scratch

import csv
import copy
import pickle
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD, Momentum, AdaGrad, Adam
from mnist_reader import load_mnist


# 조 번호 입력 (예: '6')
TEAM_NUMBER = ''
BASE_DIR = Path(__file__).resolve().parent


def load_fashion_mnist():
    """Fashion-MNIST 데이터 불러오기"""
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

    # GitHub에서 가져온 mnist_reader 사용
    x_train, t_train = load_mnist(str(data_folder), kind='train')
    x_test, t_test = load_mnist(str(data_folder), kind='t10k')
    return x_train.astype(np.float32) / 255.0, t_train, x_test.astype(np.float32) / 255.0, t_test


def train_network(hidden, lr, decay, x_train, t_train, x_val, t_val, epochs=3, optimizer_name='SGD'):
    """오차역전파로 신경망 학습"""
    np.random.seed(7)  # 비교할 때 랜덤값 고정
    net = MultiLayerNet(784, hidden, 10, weight_init_std='he', weight_decay_lambda=decay)
    losses, train_accs, val_accs = [], [], []
    best_acc, best_net = -1, None
    batch_size = 100
    optimizer_classes = {'SGD': SGD, 'Momentum': Momentum, 'AdaGrad': AdaGrad, 'Adam': Adam}
    optimizer = optimizer_classes[optimizer_name](lr=lr)

    for epoch in range(epochs):
        order = np.random.permutation(len(x_train))
        for i in range(0, len(x_train), batch_size):
            ids = order[i:i + batch_size]
            grads = net.gradient(x_train[ids], t_train[ids])
            optimizer.update(net.params, grads)

        # 학습 정확도는 2,000장만 확인
        train_acc = net.accuracy(x_train[:2000], t_train[:2000])
        val_acc = net.accuracy(x_val, t_val)
        loss = net.loss(x_train[:2000], t_train[:2000])
        losses.append(float(loss))
        train_accs.append(float(train_acc))
        val_accs.append(float(val_acc))
        print(f'  {optimizer_name} epoch {epoch + 1}: loss={loss:.4f}, train={train_acc:.3f}, val={val_acc:.3f}')
        if val_acc > best_acc:
            best_acc, best_net = val_acc, copy.deepcopy(net)

    return best_acc, best_net, losses, train_accs, val_accs


def parse_hidden(text):
    # 문자열을 은닉층 리스트로 변경
    return [int(value.strip()) for value in text.strip('[]').split(',')]


def main():
    x, t, x_test, t_test = load_fashion_mnist()

    # 훈련 데이터를 학습용과 검증용으로 나누기
    rng = np.random.RandomState(2026)
    ids = rng.permutation(len(t))
    val_ids, train_ids = ids[:6000], ids[6000:]
    x_train, t_train = x[train_ids], t[train_ids]
    x_val, t_val = x[val_ids], t[val_ids]
    del x, t
    print('학습:', x_train.shape, '검증:', x_val.shape, '테스트:', x_test.shape)

    # 비교 실험에는 일부 데이터만 사용
    small_x, small_t = x_train[:6000], t_train[:6000]
    small_val_x, small_val_t = x_val[:2000], t_val[:2000]
    results = []

    print('\n[1] 수동 비교: 은닉층 구조와 학습률')
    manual = [([50], 0.05), ([100], 0.05), ([100, 50], 0.05), ([100, 50], 0.10)]
    for hidden, lr in manual:
        acc, _, _, _, _ = train_network(hidden, lr, 0, small_x, small_t, small_val_x, small_val_t)
        results.append(['수동', str(hidden), lr, 0, float(acc)])
        print(' 결과:', hidden, lr, '검증 정확도', round(float(acc), 4))

    # 수동 실험 중 가장 좋은 구조 선택
    best_manual = max(results, key=lambda row: row[4])
    hidden = parse_hidden(best_manual[1])

    # 최적화 알고리즘 4가지 비교
    print('\n[2] SGD, Momentum, AdaGrad, Adam 비교')
    optimizer_names = ['SGD', 'Momentum', 'AdaGrad', 'Adam']
    optimizer_results = []
    optimizer_history = []
    for name in optimizer_names:
        score, _, losses, train_accs, val_accs = train_network(
            hidden, 0.01, 0, small_x, small_t, small_val_x, small_val_t,
            epochs=3, optimizer_name=name
        )
        best_epoch = int(np.argmax(val_accs)) + 1
        optimizer_results.append([
            name, float(score), best_epoch,
            float(train_accs[best_epoch - 1]),
            float(losses[best_epoch - 1]),
            float(val_accs[-1])
        ])
        for epoch, (loss, train_acc, val_acc) in enumerate(
            zip(losses, train_accs, val_accs), start=1
        ):
            optimizer_history.append([name, epoch, loss, train_acc, val_acc])
        print(f'  {name} 최고 검증 정확도: {score:.4f}')

    # 가장 높은 검증 정확도를 기록한 순서대로 결과 출력
    optimizer_ranking = sorted(optimizer_results, key=lambda row: row[1], reverse=True)
    best_optimizer = optimizer_ranking[0][0]

    print('\n========== 최적화 알고리즘 비교 결과 ==========')
    print(f'{"순위":<5} {"알고리즘":<10} {"최고 검증 정확도":>17} {"Epoch":>7} {"학습 정확도":>13} {"손실":>10}')
    print('-' * 75)
    for rank, (name, score, epoch, train_acc, loss, last_val) in enumerate(optimizer_ranking, start=1):
        print(f'{rank:<5} {name:<10} {score * 100:>16.2f}% {epoch:>7} {train_acc * 100:>12.2f}% {loss:>10.4f}')

    winner = optimizer_ranking[0]
    runner_up = optimizer_ranking[1]
    gap = (winner[1] - runner_up[1]) * 100
    print('-' * 75)
    print(f'가장 좋은 알고리즘: {winner[0]} (최고 검증 정확도 {winner[1] * 100:.2f}%)')
    print(f'2위 {runner_up[0]}보다 {gap:.2f}%p 높음')
    print('※ 비교 조건: 학습 6,000장 / 검증 2,000장 / 3 Epoch / 학습률 0.01')
    print('===============================================')

    print('\n[3] Randomized Search: 학습률과 가중치 감소를 4번 무작위 탐색')
    random_rng = np.random.RandomState(42)
    for _ in range(4):
        lr = float(10 ** random_rng.uniform(-2.0, -0.8))
        decay = float(10 ** random_rng.uniform(-6.0, -3.0))
        acc, _, _, _, _ = train_network(hidden, lr, decay, small_x, small_t, small_val_x, small_val_t, optimizer_name=best_optimizer)
        results.append(['랜덤', str(hidden), lr, decay, float(acc)])
        print(' 결과:', round(lr, 5), round(decay, 6), '검증 정확도', round(float(acc), 4))

    out = BASE_DIR / 'Competition'
    out.mkdir(exist_ok=True)

    with (out / 'optimizer_results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Optimizer', 'Best validation accuracy', 'Best epoch',
            'Train accuracy at best epoch', 'Loss at best epoch',
            'Validation accuracy at last epoch'
        ])
        writer.writerows(optimizer_ranking)
    with (out / 'optimizer_history.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Optimizer', 'Epoch', 'Loss', 'Train accuracy', 'Validation accuracy'])
        writer.writerows(optimizer_history)

    plt.figure()
    for name in optimizer_names:
        rows = [row for row in optimizer_history if row[0] == name]
        plt.plot([row[1] for row in rows], [row[4] for row in rows], marker='o', label=name)
    plt.title('Optimizer comparison - validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks([1, 2, 3])
    plt.legend()
    plt.savefig(out / 'optimizer_accuracy.png')

    plt.figure()
    for name in optimizer_names:
        rows = [row for row in optimizer_history if row[0] == name]
        plt.plot([row[1] for row in rows], [row[2] for row in rows], marker='o', label=name)
    plt.title('Optimizer comparison - training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks([1, 2, 3])
    plt.legend()
    plt.savefig(out / 'optimizer_loss.png')
    with (out / 'experiment_results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['방식', '은닉층', '학습률', '가중치감소', '검증정확도'])
        writer.writerows(results)

    # 선택된 옵티마이저로 진행한 랜덤 탐색 결과 중 최고 설정 사용
    best = max(results[len(manual):], key=lambda row: row[4])
    final_hidden, final_lr, final_decay = parse_hidden(best[1]), best[2], best[3]
    print('\n[4] 최종 모델 학습:', best_optimizer, best[:4])
    acc, network, losses, train_accs, val_accs = train_network(
        final_hidden, final_lr, final_decay,
        x_train, t_train, x_val, t_val, epochs=8, optimizer_name=best_optimizer
    )

    plt.figure()
    plt.plot(losses)
    plt.title('Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(out / 'loss.png')

    plt.figure()
    plt.plot(train_accs, label='train')
    plt.plot(val_accs, label='validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(out / 'accuracy.png')

    # 마지막에 테스트 정확도 확인
    test_acc = network.accuracy(x_test, t_test)
    print('\n========== 최종 모델 선정 결과 ==========')
    print('선택된 은닉층:', final_hidden)
    print('선택된 최적화 알고리즘:', best_optimizer)
    print(f'알고리즘 비교 최고 검증 정확도: {winner[1] * 100:.2f}%')
    print(f'선택된 학습률: {final_lr:.6f}')
    print(f'선택된 가중치 감소 계수: {final_decay:.8f}')
    print(f'최종 최고 검증 정확도: {acc * 100:.2f}%')
    print(f'최종 테스트 정확도: {test_acc * 100:.2f}%')
    print('========================================')

    filename = f'network_Team{TEAM_NUMBER}.pkl' if TEAM_NUMBER else 'network.pkl'
    with (out / filename).open('wb') as f:
        pickle.dump(network, f)
    print('저장 완료:', out / filename)

    # 학습이 끝나면 두 그래프를 화면에 표시
    plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
