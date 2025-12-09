# HDF5 직접 로딩 방식 (메모리 효율적)

이미지를 미리 추출하지 않고, 학습 시 HDF5에서 직접 읽어오는 방식입니다.

## 장점
- ✅ 빠른 준비: 메타데이터만 추출하므로 몇 초면 완료
- ✅ 디스크 절약: 이미지를 중복 저장하지 않음
- ✅ 메모리 효율적: 필요한 데이터만 로드
- ✅ 유연성: 다양한 observation key 선택 가능

## 사용 방법

### 1단계: JSON 인덱스 생성 (한 번만 실행)

```bash
# libero_10 전체 인덱스 생성
python create_dataset_index.py \
    --data-dir /mnt/data/libero/libero_10 \
    --output ./libero_10_index.json

# libero_90 전체 인덱스 생성
python create_dataset_index.py \
    --data-dir /mnt/data/libero/libero_90 \
    --output ./libero_90_index.json

# 특정 파일들만 인덱스 생성
python create_dataset_index.py \
    --datasets /mnt/data/libero/task1.hdf5 /mnt/data/libero/task2.hdf5 \
    --output ./my_tasks_index.json
```

출력 예시:
```
Processing 10 HDF5 files...
100%|████████████████████████████| 10/10 [00:05<00:00,  1.85it/s]

================================================================================
Dataset Index Created: ./libero_10_index.json
================================================================================
Total HDF5 files: 10
Total demonstrations: 500
Total frames: 132980

Tasks:
  - KITCHEN_SCENE1_task1: 50 demos, 13298 frames
  - KITCHEN_SCENE2_task2: 50 demos, 13298 frames
  ...
================================================================================
```

### 2단계: DataLoader 테스트

```bash
python example_hdf5_dataloader.py --index ./libero_10_index.json
```

### 3단계: 학습 코드에서 사용

```python
import torch
from torch.utils.data import DataLoader
from example_hdf5_dataloader import LIBEROHdf5Dataset

# Dataset 생성
dataset = LIBEROHdf5Dataset(
    index_path='./libero_10_index.json',
    obs_key='agentview_rgb',  # 또는 'eye_in_hand_rgb'
    cache_hdf5=True  # HDF5 파일 핸들 캐싱 (더 빠름)
)

# DataLoader 생성
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # 병렬 로딩
    pin_memory=True
)

# 학습 루프
for batch in dataloader:
    images = batch['image']  # (B, 3, 128, 128)
    actions = batch['action']  # (B, 7)
    language = batch['language_instruction']  # List[str]
    
    # Your training code here
    ...
```

## JSON 인덱스 구조

```json
{
  "/absolute/path/to/dataset.hdf5": {
    "file_path": "/absolute/path/to/dataset.hdf5",
    "file_name": "dataset.hdf5",
    "task_name": "KITCHEN_SCENE3_turn_on_stove",
    "num_demos": 50,
    "total_frames": 13298,
    "demos": {
      "demo_0": {
        "language_instruction": "turn on the stove and put the moka pot on it",
        "num_frames": 272,
        "action_shape": [272, 7],
        "action_dtype": "float64",
        "obs_keys": {
          "agentview_rgb": {
            "shape": [272, 128, 128, 3],
            "dtype": "uint8"
          },
          "eye_in_hand_rgb": {
            "shape": [272, 128, 128, 3],
            "dtype": "uint8"
          },
          "ee_pos": {
            "shape": [272, 3],
            "dtype": "float64"
          }
        }
      }
    }
  }
}
```

## 고급 기능

### 1. 여러 observation key 사용

```python
# eye_in_hand 카메라 사용
dataset = LIBEROHdf5Dataset(
    index_path='./libero_10_index.json',
    obs_key='eye_in_hand_rgb'
)
```

### 2. Multi-task 학습

```python
from example_hdf5_dataloader import LIBEROMultiTaskDataset

# 태스크별 가중치 설정
task_weights = {
    'task1': 2.0,  # 2배 많이 샘플링
    'task2': 1.0,  # 기본
    'task3': 0.5   # 절반만 샘플링
}

dataset = LIBEROMultiTaskDataset(
    index_path='./libero_10_index.json',
    task_weights=task_weights
)
```

### 3. 특정 태스크만 필터링

```python
import json

# 인덱스 로드
with open('./libero_10_index.json', 'r') as f:
    full_index = json.load(f)

# 원하는 태스크만 선택
filtered_index = {
    k: v for k, v in full_index.items() 
    if 'turn_on_stove' in v['task_name']
}

# 필터링된 인덱스 저장
with open('./filtered_index.json', 'w') as f:
    json.dump(filtered_index, f, indent=2)

# 사용
dataset = LIBEROHdf5Dataset('./filtered_index.json')
```

### 4. 커스텀 데이터 변환

```python
class MyCustomDataset(LIBEROHdf5Dataset):
    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        
        # 추가 변환 적용
        image = batch['image']
        
        # 예: Data augmentation
        if self.training:
            image = self.random_crop(image)
            image = self.color_jitter(image)
        
        batch['image'] = image
        return batch
```

## 성능 최적화 팁

### 1. HDF5 캐싱

```python
# 캐싱 ON (권장): 더 빠르지만 메모리 사용 증가
dataset = LIBEROHdf5Dataset(cache_hdf5=True)

# 캐싱 OFF: 느리지만 메모리 절약
dataset = LIBEROHdf5Dataset(cache_hdf5=False)
```

### 2. DataLoader workers

```python
# CPU 코어 수에 맞춰 조정
dataloader = DataLoader(
    dataset,
    num_workers=8,  # 많을수록 빠르지만 메모리 증가
    persistent_workers=True,  # worker 재사용
    prefetch_factor=2  # 미리 로드할 배치 수
)
```

### 3. SSD 사용

HDF5 파일을 SSD에 저장하면 훨씬 빠릅니다:
```bash
# HDD에서 SSD로 복사
cp -r /mnt/data/libero /ssd/libero
python create_dataset_index.py --data-dir /ssd/libero/libero_10 --output ./index.json
```

## 기존 방식과 비교

### 기존 방식 (PNG/numpy 추출)
```bash
# 모든 이미지 추출 (오래 걸림)
python extract_training_data.py --dataset task.hdf5 --output ./data
# ❌ 시간: ~10분
# ❌ 디스크: ~50GB
# ✅ 학습 속도: 빠름
```

### 새로운 방식 (HDF5 직접 로딩)
```bash
# 메타데이터만 추출 (빠름)
python create_dataset_index.py --data-dir /data --output ./index.json
# ✅ 시간: ~5초
# ✅ 디스크: ~100KB
# ✅ 학습 속도: 충분히 빠름 (특히 SSD 사용 시)
```

## Troubleshooting

### "HDF5 file not found" 에러
- JSON 인덱스에 절대 경로가 저장되므로, HDF5 파일 위치가 변경되면 인덱스를 다시 생성해야 합니다.

### 느린 학습 속도
- `cache_hdf5=True` 사용
- `num_workers` 증가
- HDF5 파일을 SSD로 이동
- `prefetch_factor` 증가

### 메모리 부족
- `cache_hdf5=False` 사용
- `num_workers` 감소
- `batch_size` 감소

## 예제 전체 학습 코드

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from example_hdf5_dataloader import LIBEROHdf5Dataset

# Dataset & DataLoader
dataset = LIBEROHdf5Dataset('./libero_10_index.json')
train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

# Model
class RobotPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Your model here
        
    def forward(self, image):
        # Predict action
        return action

model = RobotPolicy().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Training
for epoch in range(100):
    for batch in train_loader:
        images = batch['image'].cuda()
        actions = batch['action'].cuda()
        
        pred_actions = model(images)
        loss = criterion(pred_actions, actions)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```




























