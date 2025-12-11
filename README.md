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
    
결정 트리(Decision Tree)

<img width="778" height="267" alt="스크린샷 2025-11-16 오후 9 08 00" src="https://github.com/user-attachments/assets/3d855bda-dbf5-4059-8de7-f293a1277738" />

    - 매우 쉽고 유연하게 적용될 수 있는 알고리즘으로서 데이터의 스케일링, 정규화 등의 데이터 전처리의 의존도가 매우 적다.
    - 학습을 통해 데이터에 있는 규칙을 자동으로 찾아내서 Tree기반의 분류 규칙을 만든다.
    - 각 특성이 개별적으로 처리되어 데이터를 분할하는데 데이터 스케일의 영향을 받지 않으므로 결정트리에서는 정규화나 표준화같은 전처리 과정이 필요없다.
    - 영향을 가장 많이 미치는 feature를 찾아낼 수도 있다.
    - 예측 성능을 계속해서 향상시키면 복잡한 규칙 구조를 가지기 때문에 ※과적합(Overfitting)이 발생해서 예측 성능이 저하될 수도 있다.
    - 가장 상위 노드를 "루트 노드"라고 하며, 나머지 분기점을 "서브 노드", 결정된 분류값 노드를 "리프 노드"라고 한다.
    - 복잡도를 감소시키는 것이 주목적이며, 정보의 복잡도를 불순도(Impurity)라고 한다.
    - 이를 수치화한 값으로 지니 계수(Gini coeficient)가 있다.
    - 클래스가 섞이지 않고 분류가 잘 되었다면, 불순도 낮다.
    - 클래스가 많이 섞여 있고 분류가 잘 안되었다면, 불순도 높다.
    - 통계적 분산 정도를 정량화하여 표현한 값이고, 0과 1사이의 값을 가진다.
    - 지니 계수가 낮을 수록 분류가 잘 된 것이다.

과적합
    
    - 학습 데이터를 과하게 학습시켜서 실제 데이터에서는 오차가 오히려 증가하는 현상이다.

<img width="363" height="234" alt="스크린샷 2025-11-16 오후 9 10 07" src="https://github.com/user-attachments/assets/60714c89-3c40-41fe-aeda-cc3ebddf97dc" />

Graphviz

    - 결정트리 모델을 시각화할 수 있다.
    - pip install graphviz

## 📝 실습 (Lung Cancer - Dataset)

- 정규화

        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        
        gender_encoder = LabelEncoder()
        genders = gender_encoder.fit_transform(c_df.GENDER.tolist())
        c_df['GENDER'] = genders
        
        lung_cancer_encoder = LabelEncoder()
        targets = lung_cancer_encoder.fit_transform(c_df.LUNG_CANCER.tolist())
        c_df['LUNG_CANCER'] = targets


inverse_transform 

    - 정규화나 변환을 했던 걸 다시 원래 값으로 되돌리는 함수

연산 결과를 파일로 내보내기 

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.tree import export_graphviz
    import graphviz
    
    dtc_cancer = DecisionTreeClassifier()
    
    features, target = c_df.iloc[:, :-1], c_df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state = 124)
    
    dtc_cancer.fit(X_train, y_train)
    
    export_graphviz(
        dtc_cancer,
        out_file='./images/cancer_dtc01.dot',
        class_names= lung_cancer_encoder.classes_,
        feature_names= features.columns,
        impurity=True,
        filled=True
        
    )

    with open('./images/cancer_dtc01.dot') as f:
        cancer_dtc01 = f.read()
        
    cancer_dtc01_graph = graphviz.Source(cancer_dtc01)
    cancer_dtc01_graph.render(filename='cancer_dtc01', directory='./images/', format='png')


<img width="1710" height="756" alt="스크린샷 2025-11-16 오후 9 23 48" src="https://github.com/user-attachments/assets/383fa2bc-25ee-48fb-88a1-a1b68ffa7e4a" />

데이터 시각화 

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.barplot(x=dtc_cancer.feature_importances_, y= features.columns)
    plt.show()

<img width="872" height="412" alt="스크린샷 2025-11-16 오후 9 24 37" src="https://github.com/user-attachments/assets/db8604f6-916e-45b6-8deb-38217c91f579" />


Classifier의 Decision Boundary를 시각화 하는 함수
    
    import numpy as np
    
    def visualize_boundary(model, X, y):
        fig,ax = plt.subplots()
        
        # 학습 데이타 scatter plot으로 나타내기
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=25, cmap='rainbow', edgecolor='k',
                   clim=(y.min(), y.max()), zorder=3)
        ax.axis('tight')
        ax.axis('off')
        xlim_start , xlim_end = ax.get_xlim()
        ylim_start , ylim_end = ax.get_ylim()
        
        # 호출 파라미터로 들어온 training 데이타로 model 학습 . 
        model.fit(X.values, y)
        # meshgrid 형태인 모든 좌표값으로 예측 수행. 
        xx, yy = np.meshgrid(np.linspace(xlim_start,xlim_end, num=200),np.linspace(ylim_start,ylim_end, num=200))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        # contourf() 를 이용하여 class boundary 를 visualization 수행. 
        n_classes = len(np.unique(y))
        contours = ax.contourf(xx, yy, Z, alpha=0.3,
                               levels=np.arange(n_classes + 1) - 0.5,
                               cmap='rainbow',
                               zorder=1)

<img width="529" height="377" alt="스크린샷 2025-11-16 오후 9 25 59" src="https://github.com/user-attachments/assets/461f089a-5c5a-42d9-bdc4-10515375dc03" />

평가 점수 

    from sklearn.metrics import accuracy_score
    
    accuracy_score(y_test, dtc_cancer.predict(X_test[['SWALLOWING DIFFICULTY', 'AGE']].values))
        
    0.8870967741935484

베이즈 추론, 베이즈 정리, 베이즈 추정(Bayesian Inference)

    - 역확률(inverse probability) 문제를 해결하기 위한 방법으로서, 조건부 확률(P(B|A)))을 알고 있을 때, 정반대인 조건부 확률(P(A|B))을 구하는 방법이다.
    - 추론 대상의 사전 확률과 추가적인 정보를 기반으로 해당 대상의 "사후 확률"을 추론하는 통계적 방법이다.
    - 어떤 사건이 서로 "배반"하는(독립하는) 원인 둘에 의해 일어난다고 하면, 실제 사건이 일어났을 때 이 사건이 두 원인 중 하나일 확률을 구하는 방식이다.
    - 어떤 상황에서 N개의 원인이 있을 때, 실제 사건이 발생하면 N개 중 한 가지 원인일 확률을 구하는 방법이다.
    - 기존 사건들의 확률을 알 수 없을 때, 전혀 사용할 수 없는 방식이다.
    - 하지만, 그 간 데이터가 쌓이면서, 기존 사건들의 확률을 대략적으로 뽑아낼 수 있게 되었다.
    - 이로 인해, 사회적 통계나 주식에서 베이즈 정리 활용이 필수로 꼽히고 있다.

나이브 베이즈 분류 (Naive Bayes Classifier)

    - 텍스트 분류를 위해 전통적으로 사용되는 분류기로서, 분류에 있어서는 준수한 성능을 보인다.
    - 베이즈 정리에 기반한 통계적 분유 기법으로서, 정확성도 높고 대용량 데이터에 대한 속도도 빠르다.
    - 반드시 모든 feature가 서로 독립적이어야 한다. 즉, 서로 영향을 미치지 않는 feature들로 구성되어야 한다.
    - 감정 분석, 스팸 메일 필터링, 텍스트 분류, 추천 시스템 등 여러 서비스에서 활용되는 분류 기법이다.
    - 빠르고 정확하고 간단한 분류 방법이지만, 실제 데이터에서 모든 feature가 독립적인 경우는 드물기 때문에 실생활에 적용하기 어렵가.

CountVectorizer

    - 문장에 있는 단어들에 인덱스를 붙여서 각 단어의 빈도수를 세어주는 기술
    - from sklearn.feature_extraction.text import CountVectorizer















