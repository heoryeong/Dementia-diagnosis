import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# CSV 불러오기
df = pd.read_csv("C:/osp_pakin/train_activity.csv.part0")

# 정규화할 열
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

# Z-score 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[columns_to_scale])

# 실루엣 점수를 저장할 리스트
silhouette_scores = []
k_values = range(2, 11)  # k=2부터 10까지 확인

# k를 바꿔가며 클러스터링 수행
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"k = {k}, 실루엣 점수 = {score:.4f}")

# 결과 시각화
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Silhouette score by K")
plt.xlabel("cluster num (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()
