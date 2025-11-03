# Encoder Comparison Experiment Plan

## 목표
DROID 데이터셋에서 세 가지 encoder (VAE, VQ-VAE, IDM-Transformer)를 공정하게 비교하여 
action prediction에 가장 적합한 encoder를 찾기

## 실험 1: Reconstruction Quality Test

### 1.1 VAE (Vanilla)
```bash
# Train
python train_ddp.py --config config_vae_droid.yaml

# Test reconstruction
python test_reconstruction.py --model vae --checkpoint logs_vae/VanillaVAE_droid/checkpoints/vae_final.pt
```

**측정 항목:**
- Reconstruction MSE
- PSNR, SSIM, LPIPS
- Latent dimension: 64 vs 256 비교
- Training time per epoch

### 1.2 VQ-VAE (LAPA-style)
```bash
# Train  
python train_vqvae.py --config config_vqvae_droid.yaml

# Test reconstruction
python test_reconstruction.py --model vqvae --checkpoint logs_vqvae/checkpoints/vqvae_final.pt
```

**측정 항목:**
- Reconstruction MSE
- Codebook usage (perplexity)
- Codebook size: 8 vs 16 vs 32 비교
- Latent dim: 32

### 1.3 IDM-Transformer (UniSkill-style)
```bash
# Train
python train_idm.py --config config_idm_droid.yaml

# Test reconstruction  
python test_reconstruction.py --model idm --checkpoint logs_idm/checkpoints/idm_final.pt
```

**측정 항목:**
- Reconstruction MSE
- Attention weights visualization
- Latent dim: 64

---

## 실험 2: Action Prediction Test (핵심!)

### 2.1 Setup
각 encoder의 latent를 사용하여 **action predictor** 학습:

```
Input: latent(frame_t), latent(frame_{t+k})
Output: predicted action
Ground truth: actual robot action (if available) or predicted action from dynamics model
```

### 2.2 실험 구성

#### Option A: Action Reconstruction (if ground truth action available)
```python
# Encoder는 freeze, Action MLP만 학습
action_predictor = MLP(
    input_dim=latent_dim * 2,  # concat current & next latent
    hidden_dims=[256, 128],
    output_dim=action_dim,  # e.g., 7D robot action
)

loss = MSE(predicted_action, ground_truth_action)
```

#### Option B: Frame Prediction (더 실용적!)
```python
# Encoder로 current frame를 encode
# Action predictor로 action을 예측
# Decoder로 next frame을 복원
# 복원된 frame과 실제 next frame 비교

pipeline:
  z_curr = encoder(frame_t)
  action_pred = action_predictor(z_curr, z_next_target)
  z_next_pred = dynamics_model(z_curr, action_pred)
  frame_next_pred = decoder(z_next_pred)
  
loss = MSE(frame_next_pred, frame_{t+k})
```

### 2.3 평가 메트릭
```python
metrics = {
    # Quantitative
    'frame_prediction_mse': 'MSE between predicted and actual next frame',
    'frame_prediction_psnr': 'PSNR',
    'frame_prediction_ssim': 'SSIM',
    
    # Qualitative  
    'visual_quality': 'Human evaluation of predicted frames',
    'temporal_consistency': 'Smoothness across predicted sequence',
    
    # Latent space quality
    'latent_smoothness': 'L2 distance between consecutive latents',
    'latent_interpretability': 'Can we decode meaningful actions?',
}
```

---

## 실험 3: Ablation Studies

### 3.1 Latent Dimension Ablation
각 모델에서 latent dimension 변화 실험:

| Model | Latent Dims to Test |
|-------|---------------------|
| VAE | 32, 64, 128, 256 |
| VQ-VAE | 32 (fixed), codebook_size: 8, 16, 32, 64 |
| IDM | 64, 128, 256 |

### 3.2 Architecture Ablation
- VAE: hidden_dims 깊이 변화
- VQ-VAE: codebook size, commitment loss weight
- IDM: transformer layers, attention heads

### 3.3 Training Data Size
- 100 episodes (debug)
- 1K episodes  
- 10K episodes
- Full dataset

---

## 실험 4: 실전 테스트 (Optional, but powerful!)

만약 시뮬레이터나 실제 로봇이 있다면:

### 4.1 Policy Learning Test
```python
# Train policy with frozen encoder
policy = Policy(encoder, action_head)
encoder.freeze()

# Train on demonstration data
# Test on real robot tasks
success_rate = evaluate_on_real_tasks(policy)
```

### 4.2 Few-shot Adaptation
```python
# Encoder 학습 후 새로운 task에 빠르게 적응 가능한가?
# 10 demos로 fine-tune
# Success rate 측정
```

---

## 타임라인

### Week 1: Reconstruction Experiments
- Day 1-2: VAE 학습 및 평가
- Day 3-4: VQ-VAE 구현 및 학습
- Day 5-6: IDM 구현 및 학습
- Day 7: Reconstruction 결과 비교 및 분석

### Week 2: Action Prediction Experiments  
- Day 1-3: Action prediction framework 구현
- Day 4-5: 각 encoder로 action prediction 학습
- Day 6-7: 결과 비교 및 분석

### Week 3: Ablation & Analysis
- Day 1-3: Latent dimension ablation
- Day 4-5: Architecture ablation
- Day 6-7: 최종 분석 및 논문/보고서 작성

---

## 예상 결과 (가설)

### VAE
- **장점**: Smooth continuous latent space, 보간 가능
- **단점**: Posterior collapse 위험, KLD 튜닝 필요
- **예상**: Mid-range performance, stable training

### VQ-VAE (LAPA-style)
- **장점**: Discrete, interpretable, no posterior collapse
- **단점**: Codebook collapse 위험, quantization error
- **예상**: Best reconstruction, good for discrete actions

### IDM (UniSkill-style)
- **장점**: Transformer의 long-range dependencies
- **단점**: 더 많은 데이터 필요, 학습 느림
- **예상**: Best for complex temporal patterns

---

## 코드 구조

```
vae_latent/
├── models/
│   ├── vanilla_vae.py          # 현재 VAE
│   ├── vq_vae.py                # 새로 구현
│   └── idm_transformer.py       # 새로 구현
├── train_vae.py                 # VAE 학습
├── train_vqvae.py              # VQ-VAE 학습
├── train_idm.py                # IDM 학습
├── test_reconstruction.py       # Reconstruction 평가
├── test_action_prediction.py    # Action prediction 평가
├── compare_encoders.py          # 종합 비교
└── configs/
    ├── vae_*.yaml
    ├── vqvae_*.yaml
    └── idm_*.yaml
```

---

## 논문/보고서 구성

1. **Introduction**: Action representation learning의 중요성
2. **Related Work**: VAE, VQ-VAE, Transformer encoders
3. **Methods**: 세 가지 encoder 설명
4. **Experiments**: 
   - Reconstruction quality
   - Action prediction performance
   - Ablation studies
5. **Results**: 정량적/정성적 비교
6. **Discussion**: 각 방법의 trade-offs
7. **Conclusion**: Best encoder recommendation

---

## 추가 고려사항

### Computational Resources
- GPU memory: VQ-VAE < VAE < IDM-Transformer
- Training time: VAE < VQ-VAE < IDM-Transformer
- Inference speed: 모두 비슷 (single forward pass)

### Implementation Complexity
- VAE: ⭐⭐ (이미 완성)
- VQ-VAE: ⭐⭐⭐ (NSVQ 참고하여 구현)
- IDM-Transformer: ⭐⭐⭐⭐ (가장 복잡)

### Data Requirements
- VAE: Medium (continuous space는 적은 데이터로도 가능)
- VQ-VAE: Medium-High (codebook 학습 필요)
- IDM-Transformer: High (attention 학습에 많은 데이터 필요)








