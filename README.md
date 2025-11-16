#### 분류 (Classifier)

    - 대표적인 지도학습 방법 중 하나이며, 다양한 문제와 정답을 학습한 뒤 별도의 테스트에서 정답을 예측한다.
    - 주어진 문제와 정답을 먼저 학습한 뒤 새로운 문제에 대한 정답을 예측하는 방식이다.
    - 이진 분류 (Binary Classification)의 경우 정답은 0(Negative)과 1(Positive)과 같이 True, False값을 가진다.
    - 다중 분류 (Muticlass Classification)는 정답이 가질 수 있는 값은 3개 이상이다(예: 0, 1, 2, 3).


#### Feature 

    - 데이터 세트의 일반 컬럼이며, 2차원 이상의 다차원 데이터까지 통틀어 피처라고 한다.
    - 타켓을 제외한 나머지 속성을 의미한다.

<img width="765" height="270" alt="스크린샷 2025-11-16 오후 8 56 28" src="https://github.com/user-attachments/assets/849cb23f-d21a-4f14-a9fe-3dfc48c81fc7" />

#### 분류 예측 프로세스 

<img width="1115" height="622" alt="스크린샷 2025-11-16 오후 8 57 05" src="https://github.com/user-attachments/assets/fe5fc670-ec7b-4e17-aa34-b7f5bf7ae163" />

#### scikit-learn

    - 파이썬 머신러닝 라이브러리
    - 데이터만 주면 예측/분류/군집/정규화/모델평가까지 다 해준다.
    
      pip install scikit-learn

데이터 세트 분리

    train_test_split(feature, target, test_size, random_state)
    - 학습 데이터 세트와 테스트 데이터 세트를 분리해준다.
    - feature: 전체 데이터 세트 중 feature
    - target: 전체 데이터 세트 중 target
    - test_size: 테스트 세트의 비율 (0 ~ 1)
    - random_state: 매번 동일한 결과를 원할 때, 원하는 seed(기준점)를 작성한다.

모델학습 

    fit(train_feature, train_target)
    
    - 모델을 학습시킬 때 사용한다.
    - train_feature: 훈련 데이터 세트 중 feature
    - train_target: 훈련 데이터 세트중 target

평가 

    accuracy_score(y_test, predict(X_test))
    
    - 모델이 얼마나 잘 얘측했는지를 '정확도'라는 평가 지표로 평가할 때 사용한다.
    - y_test: 실제 정답
    - predict(X_test): 예측한 정답
    
















