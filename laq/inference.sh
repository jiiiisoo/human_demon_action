python inference_openx.py \
    --input_file /workspace/LAPA/laq/test_sample.jsonl \
    --laq_checkpoint /workspace/LAPA/lapa_checkpoints/laq_openx.pt \
    --output_file my_latent_actions.jsonl \
    --batch_size 8 \
    --image_size 256 \
    --window_size 2 \
    --save_images \
    --output_dir /workspace/LAPA/laq/test_sample_images