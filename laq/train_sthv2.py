from laq_model import LAQTrainer
from laq_model import LatentActionQuantization



laq = LatentActionQuantization(
    dim = 1024,
    quant_dim=32,
    codebook_size = 8,
    image_size = 256,
    patch_size = 32,
    spatial_depth = 8, #8
    temporal_depth = 8, #8
    dim_head = 64,
    heads = 16,
    code_seq_len=4,
).cuda()


trainer = LAQTrainer(
    laq,
    folder = '/workspace/dataset/something_to_something/frames_full',
    offsets = 30,
    batch_size = 100,
    grad_accum_every = 1,
    train_on_images = False, 
    use_ema = False,          
    num_train_steps = 100005,  # Reduced for testing
    results_folder='results',
    lr=1e-4,
    save_model_every=5000,
    save_results_every=5000,
    wandb_project='latent_action',
    wandb_run_name='base_lapa_sthv2',
    use_wandb=False,  # Set to False to disable wandb
    use_tensorboard=True,
    tensorboard_log_dir='./runs',
)

trainer.train()        

