import numpy as np
import pandas as pd

# 임의의 사용자 샘플 (레벨 0, 1, 2, 3)
sample_data = pd.DataFrame([
    # 레벨 0 (저위험군): 매우 활발한 활동자
    {
        'activity_score': 0.1,
        'activity_steps': -0.2,
        'activity_inactive': 0.3,
        'activity_high': 0.1,
        'activity_score_meet_daily_targets': -0.1,
        'activity_score_move_every_hour': -0.2,
        'activity_daily_movement': 0.0,
        'activity_total': -0.1
    },
    # 레벨 1 (중하 위험군): 평균보다는 살짝 낮은 활동
    {
        'activity_score': 1.8,
        'activity_steps': 2.0,
        'activity_inactive': -1.5,
        'activity_high': 1.8,
        'activity_score_meet_daily_targets': 2.0,
        'activity_score_move_every_hour': 1.5,
        'activity_daily_movement': 1.9,
        'activity_total': 2.2
    },
    # 레벨 2 (중상 위험군)
    {
        'activity_score': -2.2,
        'activity_steps': -2.3,
        'activity_inactive': 2.4,
        'activity_high': -2.1,
        'activity_score_meet_daily_targets': -2.0,
        'activity_score_move_every_hour': -2.1,
        'activity_daily_movement': -2.2,
        'activity_total': -2.5
    },
    # 레벨 3 (고위험군): 거의 움직임 없는 상태
    {
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

print(sample_data)
