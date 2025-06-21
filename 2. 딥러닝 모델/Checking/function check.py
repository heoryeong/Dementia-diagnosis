import torch
import torch.nn as nn
import pandas as pd

# 1. MLPClassifier 클래스 정의 (최종 학습 때와 동일)
class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)

# 2. 학습된 모델 로드
model = MLPClassifier()
model.load_state_dict(torch.load("C:/osp_pakin/final_model.pth"))
model.eval()

# 3. 샘플 사용자 데이터 (Z-score 정규화 기준, 직접 추정한 예시 데이터, 필요할 때 하나씩 가져와서 쓰기)
sample_data = pd.DataFrame([{
    #레벨3
        'activity_score': -2.1,
        'activity_steps': -2.4,
        'activity_inactive': 2.3,
        'activity_high': -2.0,
        'activity_score_meet_daily_targets': -2.1,
        'activity_score_move_every_hour': -2.0,
        'activity_daily_movement': -2.3,
        'activity_total': -2.4
    }
])

# 4. Tensor로 변환
input_tensor = torch.tensor(sample_data.values, dtype=torch.float32)

# 5. 예측 수행
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()

# 6. 결과 출력
print(f"사용자 예측된 위험도 레벨: {prediction}")
