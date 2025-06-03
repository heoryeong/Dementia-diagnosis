import pandas as pd
import matplotlib.pyplot as plt


# CSV 파일 불러오기
df = pd.read_csv("C:/osp_pakin/train_activity.csv.part0")  # 파일 경로는 필요에 따라 조정

# 분석할 열 지정
column = 'activity_score_meet_daily_targetsr'  # 여기에 원하는 컬럼명을 바꾸면 됨

# 결측치 제거 (필수)
data = df[column].dropna()

# Q1, Q3, IQR 계산
q1 = data.quantile(0.25)
q3 = data.quantile(0.75)
iqr = q3 - q1

# 이상치 기준 계산
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# 이상치 탐지
outliers = data[(data < lower_bound) | (data > upper_bound)]

# 이상치 행 출력
outlier_rows = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
print(outlier_rows)

# 결과 출력
print(f"[{column}] 열의 전체 값 개수 (결측치 제외): {len(data)}")
print(f"[{column}] 열 이상치 개수: {len(outliers)}")
print(f"이상치 하한: {lower_bound:.2f}, 상한: {upper_bound:.2f}")
print("이상치 예시 상위 5개:")
print(outliers.head())
print("모든 이상치:")
print(outliers)

#이상치가 없는 결과값이 이상해서 확인용으로 히스토그램
df['activity_score_meet_daily_targets'].hist(bins=30) #원하는 칼럼명으로 바꾸기
plt.title('Distribution of activity_score_meet_daily_targets') #이거는 그래프에 보여지는 용용
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
