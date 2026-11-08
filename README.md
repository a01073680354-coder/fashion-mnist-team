# Fashion-MNIST 팀 프로젝트

4·5·6장 강의내용을 바탕으로 만든 간단한 완전연결 신경망입니다.

## 코드 보기
- [신경망 학습 및 실험 코드](fashion_mnist_team.py)
- [다층 신경망 클래스](common/multi_layer_net.py)
- [계층 구현](common/layers.py)

## 실행
Git과 Python 설치 후 이 저장소를 다운로드하거나 복제하고, 다음 명령을 실행합니다.

```powershell
python -m pip install numpy matplotlib
python fashion_mnist_team.py
```

첫 실행 시 [공식 Fashion-MNIST GitHub](https://github.com/zalandoresearch/fashion-mnist)에서 의류 이미지 데이터를 복제합니다. 손글씨 숫자 MNIST와 다른 데이터셋입니다.

모델은 2층/3층 구조와 학습률을 수동으로 비교한 다음, 학습률·가중치 감소를 4회 무작위 탐색합니다. 데이터는 학습 54,000장, 검증 6,000장, 테스트 10,000장으로 나누고, 최종 모델을 선택한 이후에만 테스트 정확도를 평가합니다.

실험 결과, 학습 그래프 및 모델(.pkl)은 로컬 Competition 폴더에 저장됩니다. 실제 데이터셋 전체 학습 결과는 아직 별도로 검증되지 않았습니다.

## 참고
- [Fashion-MNIST 공식 저장소](https://github.com/zalandoresearch/fashion-mnist)
- [공식 mnist_reader.py](https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py)
- [밑바닥부터 시작하는 딥러닝 교재 코드](https://github.com/WegraLee/deep-learning-from-scratch)
