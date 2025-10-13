#!/bin/bash
# cuDNN 문제 해결 스크립트

# python3를 python으로 notation 바꾸기
alias python=python3


echo "1. 시스템 cuDNN 비활성화..."
if [ -f "/usr/lib/x86_64-linux-gnu/libcudnn.so.8.8.0" ]; then
    mv /usr/lib/x86_64-linux-gnu/libcudnn.so.8.8.0 /usr/lib/x86_64-linux-gnu/libcudnn.so.8.8.0.bak
fi

echo "2. 올바른 cuDNN 설치..."
pip install --upgrade "nvidia-cudnn-cu12==8.9.7.29"

echo "3. JAX 테스트..."
LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH" python -c "
import jax
from tux import set_random_seed
set_random_seed(1)
print('JAX cuDNN 설정 완료!')
"

echo "4. LAPA 실행 준비 완료!"
echo "실행 명령어:"
echo "cd /workspace/LAPA && LD_LIBRARY_PATH=\"/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:\$LD_LIBRARY_PATH\" CUDA_VISIBLE_DEVICES=0 python -m latent_pretraining.inference --mesh_dim \"1,1,1,1\""

pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

cd /workspace/LAPA/laq
pip install -e .