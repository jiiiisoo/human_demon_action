# export TORCHVISION_DISABLE_TORCHCODEC=1

from transformers import AutoVideoProcessor, AutoModel
import torch
# from torchcodec.decoders import VideoDecoder
import cv2
import numpy as np

hf_repo = "facebook/vjepa2-vitg-fpc64-256"

model = AutoModel.from_pretrained(hf_repo).to("cuda")
processor = AutoVideoProcessor.from_pretrained(hf_repo)

video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"
cap = cv2.VideoCapture(video_url)
frame_idx = np.arange(0, 64) # choosing some frames. here, you can define more complex sampling strategy
video = []
for i in frame_idx:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        break
    video.append(frame)
video = np.array(video)
# video = video.transpose(0, 3, 1, 2)
print(video.shape)
print(model.device)
video = processor(video, return_tensors="pt").to(model.device)
with torch.no_grad():
    video_embeddings = model.get_vision_features(**video)

print(video_embeddings.shape)