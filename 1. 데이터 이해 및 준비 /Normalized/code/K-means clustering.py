import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CSV 불러오기
df = pd.read_csv("C:/osp_pakin/train_activity.csv.part0")

#Z-score 코드에서 따옴
# 정규화할 8개 열 리스트
columns_to_scale = [
    'activity_score',
    'activity_steps',
    'activity_inactive',
    'activity_high',
    'activity_score_meet_daily_targets',
    'activity_score_move_every_hour',
    'activity_daily_movement',
    'activity_total'
]
# StandardScaler 객체 생성
scaler = StandardScaler()
# Z-score 정규화 적용
scaled_data = scaler.fit_transform(df[columns_to_scale])
# 정규화된 데이터를 DataFrame으로 다시 구성
scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale)
# 기존 df에서 정규화 안한 열은 그대로 유지하고, 정규화된 열만 덮어쓰기
for col in columns_to_scale:
    df[col] = scaled_df[col]


# 2. 사용할 열만 선택
features = [
    'activity_score',
    'activity_steps',
    'activity_inactive',
    'activity_high',
    'activity_score_meet_daily_targets',
    'activity_score_move_every_hour',
    'activity_daily_movement',
    'activity_total'
]
X = df[features].copy()

# 3. 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. KMeans 클러스터링 (k=2로 시작)
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 5. 클러스터별 통계 확인
print(df.groupby('cluster')[features].mean())

# 6. 시각화
sns.pairplot(df, hue='cluster', vars=features[:4])  # 일부만 시각화
plt.show()

# 7. 클러스터링 결과 저장
df.to_csv("C:/osp_pakin/clustered_data.csv", index=False)
