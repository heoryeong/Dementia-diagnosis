# (Main Branch)Parkinson's disease-diagnosis _ Main Branch

목적

웨어러블 기기로 수집된 걸음걸이 데이터로 치매(파킨슨) 고위험군 예측 분류 모델 만들기


메뉴얼

0. 역할 분담

  역할 분담 및 프로젝트 진행 방향에 대한 초기 계획이 있음

1. 데이터 이해 및 준비

   [Data] Folder

     걸음걸이 원천 데이터와 거기서 고른 8개의 요소만 따로 정리한 데이터 파일이 있음

   [Data_preparation_bound] Folder

     결측치와 이상치 관련 내용이 정리되어있음

   [Normalized] Folder

     전처리와 라벨링을 위한 각종 파일과 그 결과(그래프, clustered data)가 있음

     [code] Folder

       보는 순서는 아래 참고

       1. Normalized.ipynb, normalized.py
       2. shilouette_K.py
       3. K-means clustering.py
       4. SHAP labeling.py
       5. labeling_and_preprocessing.py

2. 딥러닝 모델

   clustered_data2.zip : 딥러닝 모델에서 불려와지는 전처리와 라벨링이 모두 끝난 최종 csv 파일

   Parkinson's_Deseases_Diagnosis_by_Pytorch.ipynb : 초기 MLP 모델

   final model.ipynb : 최종 모델

   [trials] Folder

     초기모델에서 최종 모델까지 만들기 까지의 실험적 과정 (언급안된 그 외 파일도 실험적 내용임)

   [Checking] Folder

     Sample_data.py : 임의의 데이터, 각 레벨마다 있음

     final_model.pth : 최종적으로 학습된 모델의 가중치 정보를 저장한 파일

     function check.py : 실제로 학습이 잘 되었는지 확인하기 위해 임의의 데이터를 넣어 점검한다


개발 내용

[우리가 가지고 있는 것]

웨어러블 기기를 통해 치매 위험군들의 걸음걸이 정보를 정리한 csv 파일


[지금까지의 과정]

데이터 구조 분석

  csv파일 구조와 변수들의 의미를 파악했다.
  
  개중 응용하여 치매 고위험군이라고 판별 내릴만한 8개의 열(기준)을 선정했다.
  'activity_score', 'activity_steps', 'activity_inactive', 'activity_high', 'activity_score_meet_daily_targets', 'activity_score_move_every_hour', 'activity_daily_movement', 'activity_total'

결측치 및 이상치 탐색

  누락값과 이상치 존재 여부를 확인하려고 했다.

  결측치(누락값)은 모든 칼럼에 없다.
  
  이상치는 8개의 칼럼마다 그 개수와 버주가 너무나 차이가 난다

    이상치를 제거하는 것이 아닌, 보완적 처리로 진행하고자 한다

      어떻게 보완할 것이냐 > K-means 클러스터링으로 그룹화 하겠다

전처리 파이프라인 설계 (Z-score 기반 정규화)

  클러스터링 작업에 앞서, 원본 데이터는 각각의 열마다 단위와 범주 간의 차이가 크다

    steps : 수천~수만, high : 0~100

    이 상태는 좋은 클러스터링 결과를 나타낼 수 없다 (steps 기준으로 분류됨)

  Z-score 정규화를 통해 모든 feature가 동등한 비중으로 클러스터링에 참여하게 만들고자 했다

통계적 특징 추출 (K-means clustering)

  클러스터링(분류) 작업에 앞서, 최적의 k값을 찾기 위해 실루엣 점수를 계산했다

  분류 0번은 정상 활동, 1번은 신체 기능 저하 의심군으로 분류되었다

-------

사전에 수행한 K-means 클러스터링 결과 (2개 군집)을 바탕으로

클러스터 0은 정상군, 클러스터 1은 신체기능 저하 및 치매 위험군으로 간주하고,

클러스터 결과를 타깃 라벨(y), 신체 활동 관련 8개 feature를 입력값(x)으로 하여

XGBoost 분류 모델 학습시킴

학습된 XGBoost 모델에 대해 SHAP 값을 적용하여,

  -각 샘플의 예측에 대해,

  -각 feature가 치매 위험군(클러스터 1)에 얼마나 영향을 주었는지를 계산함

모든 특성의 기여도를 절댓값 기준으로 합산,

한 샘플이 전체적으로 위험군에 얼마나 가까운가를 정량적으로 나타낼 수 있다

  >> 이 값을 우리는 “SHAP 총 기여도” 또는 “위험 기여도 점수”로 정의함

SHAP 총 기여도를 정규화(0~100%)로 환산 후, 위험도 등급 분류했다.

--------

모델 구조 설계

self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

입력층에서는 8개의 feature를 받도록 설정

은닉층에서는 64>32로 점진적으로 축소되며 ReLU 활성화 함수 사용

출력층은 4개의 노드, 각 클래스(0~3레벨)에 대한 확률을 산출한다

*은닉층 수와 노드 개수는 간단한 구조로 시작하여 과적합 방지를 의도했다

**상기 이유로 Dropout은 초기에는 포함하지 않고, 과적합 여부를 확인 후 추후 도입 예정


하이퍼파라미터

학습률 (lr=0.001): Adam 옵티마이저에 일반적으로 권장되는 기본값 사용

Batch Size = 64: 적절한 계산 효율성과 일반화 성능 확보 목적

Epoch 수 = 50: 초기 학습 성능을 확인하기 위한 탐색적 설정


학습 구조

각 epoch 마다 train loss와 validation loss를 추적하여 과적합 여부 실시간 확인

model.eval() 및 torch.no_grad()를 사용하여 평가 시 gradient 계산을 방지했다.



성능 평가

classification_report()

confusion_matrix()

테스트셋에서 예측된 결과를 기반으로 precision, recall, f1-score를 확인했다

confusion matrix 시각화로 특정 클래스 간 혼동 여부를 파악한다
