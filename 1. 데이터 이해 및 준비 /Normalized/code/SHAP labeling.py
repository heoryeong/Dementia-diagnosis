import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 불러오기
df = pd.read_csv("C:/osp_pakin/clustered_data.csv")

# 2. 사용할 feature 선택
features = [
    'activity_score', 'activity_steps', 'activity_inactive', 'activity_high',
    'activity_score_meet_daily_targets', 'activity_score_move_every_hour',
    'activity_daily_movement', 'activity_total'
]
X = df[features]

# 3. KMeans 클러스터링 결과
y = df['cluster']

# 4. Z-score 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. XGBoost 분류 모델 학습
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 7. SHAP 값 계산
explainer = shap.Explainer(model)
shap_values = explainer(X_scaled)

# 8. 각 샘플별 SHAP 총합(절댓값 기준) 계산 → 위험도 점수로 해석
shap_total_contrib = np.abs(shap_values.values).sum(axis=1)

# 9. 정규화하여 0~100%로 위험도 점수 환산
risk_percent = (shap_total_contrib / shap_total_contrib.max()) * 100
df['risk_percent'] = risk_percent

# 10. 위험도 등급 매핑 함수
def risk_label(percent):
    if percent <= 25:
        return 0  # 치매 저위험군
    elif percent <= 50:
        return 1  # 중하 위험군
    elif percent <= 75:
        return 2  # 중상 위험군
    else:
        return 3  # 고위험군

# 11. 등급 컬럼 생성
df['risk_level'] = df['risk_percent'].apply(risk_label)

# 12. 확인
print(df[['risk_percent', 'risk_level']].head())

# 13. 시각화 (선택)
plt.figure(figsize=(10, 6))
sns.histplot(df['risk_level'], bins=4, kde=False)
plt.title("Risk level distribution")
plt.xlabel("Risk Level (0=Low, 3=High)")
plt.ylabel("Number of Participants")
plt.show()

# 14. SHAP Summary Plot
shap.summary_plot(shap_values, features=X, feature_names=features)
