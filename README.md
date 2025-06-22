# Parkinson's disease-diagnosis

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
