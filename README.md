# Fashion-MNIST 팀플

수업에서 배운 4~6장 내용을 이용해서 옷 이미지 10종류를 구분하는 신경망을 만들었습니다.

## 파일
- `fashion_mnist_team.py`: 데이터 불러오기, 학습, 실험
- `mnist_reader.py`: GitHub에서 가져온 데이터 읽기 코드
- `common/multi_layer_net.py`: 신경망 모델
- `common/layers.py`: ReLU, Affine, Softmax 계층

## 실행 방법
Git과 Python을 설치한 뒤 터미널에서 실행합니다.

```powershell
python -m pip install numpy matplotlib
python fashion_mnist_team.py
```

처음 실행하면 [Fashion-MNIST GitHub](https://github.com/zalandoresearch/fashion-mnist)에서 데이터를 받습니다.

## 실험 내용
- 은닉층 크기와 학습률을 바꿔서 4번 비교
- SGD, Momentum, AdaGrad, Adam을 같은 조건(학습률 0.01, 3 Epoch)으로 비교
- 검증 정확도가 가장 높은 최적화 알고리즘 선택
- 선택한 알고리즘으로 Randomized Search 4번 진행
- 검증 정확도가 높은 설정으로 최종 학습
- 테스트 데이터는 마지막 성능 확인에만 사용

알고리즘 비교 결과는 터미널에 **최고 검증 정확도 순위, 1위 알고리즘, 2위와의 정확도 차이**로 출력됩니다.

비교 결과는 `Competition/optimizer_results.csv`(최고 검증 정확도, 최고 Epoch, 해당 Epoch의 학습 정확도와 손실), 에폭별 결과는 `optimizer_history.csv`에 저장됩니다. 네 알고리즘의 정확도와 손실 그래프도 같은 폴더에 저장됩니다.

전체 학습 그래프와 실험 결과도 `Competition` 폴더에 저장됩니다.
조 번호는 `fashion_mnist_team.py`의 `TEAM_NUMBER`에 입력하면 됩니다.

## 참고한 GitHub
- [Fashion-MNIST 데이터](https://github.com/zalandoresearch/fashion-mnist)
- [데이터 불러오는 코드](https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py)
- [밑바닥부터 시작하는 딥러닝](https://github.com/WegraLee/deep-learning-from-scratch)

※ 전체 학습 결과와 Competition 호환성은 아직 확인하지 않았습니다.

## AI 활용
프로젝트를 진행하면서 ChatGPT의 도움을 받았습니다. 강의자료와 GitHub 예제를 참고해 코드 구성, 주석 정리, 실험 방법을 정할 때 활용했습니다. 학습 결과와 코드 실행 여부는 직접 확인할 예정입니다.
