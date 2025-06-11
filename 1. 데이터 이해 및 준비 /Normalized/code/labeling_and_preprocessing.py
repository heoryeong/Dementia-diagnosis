import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

# 2. SHAP 결과가 포함된 라벨링된 데이터 로드
df = pd.read_csv("C:/osp_pakin/clustered_data.csv")

# 3. feature와 label 선택
features = [
    'activity_score', 'activity_steps', 'activity_inactive', 'activity_high',
    'activity_score_meet_daily_targets', 'activity_score_move_every_hour',
    'activity_daily_movement', 'activity_total'
]
X = df[features].values.astype(np.float32)
y = df['risk_level'].values.astype(np.int64)

# 4. 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

# 5. TensorDataset으로 변환
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

# 6. DataLoader 정의
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# 7. 확인
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# 8. 나중에 모델학습에서 빠르게 불러오기 위한 저장용 코드 (pythorch 전용 저장 방식)
torch.save(train_dataset, "C:/osp_pakin/train_dataset.pt")
torch.save(val_dataset, "C:/osp_pakin/val_dataset.pt")
torch.save(test_dataset, "C:/osp_pakin/test_dataset.pt")

# 혹시모르니까 csv 파일로 전환해서 저장해두기
# NumPy 배열을 DataFrame으로 변환
train_df = pd.DataFrame(X_train, columns=features)
train_df['risk_level'] = y_train

val_df = pd.DataFrame(X_val, columns=features)
val_df['risk_level'] = y_val

test_df = pd.DataFrame(X_test, columns=features)
test_df['risk_level'] = y_test

train_df.to_csv("C:/osp_pakin/train_data.csv", index=False)
val_df.to_csv("C:/osp_pakin/val_data.csv", index=False)
test_df.to_csv("C:/osp_pakin/test_data.csv", index=False)
