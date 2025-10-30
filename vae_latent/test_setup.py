"""
Test script to verify the VAE setup is working correctly.
"""
import torch
from vanilla_vae_model import VanillaVAE
from dataset_sthv2 import SthV2FrameDataset
import yaml


def test_model():
    """Test that the model can be instantiated and forward pass works."""
    print("Testing model instantiation and forward pass...")
    
    model = VanillaVAE(
        in_channels=3,
        latent_dim=256,
        hidden_dims=[64, 128, 256, 512, 1024],
        image_size=256
    )
    
    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 256, 256)
    
    with torch.no_grad():
        outputs = model(dummy_input)
        recons, input_img, mu, log_var = outputs
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Reconstruction shape: {recons.shape}")
    print(f"  Mu shape: {mu.shape}")
    print(f"  Log var shape: {log_var.shape}")
    
    # Test loss function
    loss_dict = model.loss_function(*outputs, kld_weight=0.0001)
    print(f"  Loss: {loss_dict['loss'].item():.4f}")
    print(f"  Reconstruction Loss: {loss_dict['Reconstruction_Loss'].item():.4f}")
    print(f"  KLD: {loss_dict['KLD'].item():.4f}")
    
    print("✓ Model test passed!")
    return True


def test_dataset(data_path):
    """Test that the dataset can load data."""
    print(f"\nTesting dataset loader...")
    print(f"  Data path: {data_path}")
    
    try:
        dataset = SthV2FrameDataset(
            data_path,
            image_size=256,
            split='train',
            val_split=0.05
        )
        
        print(f"  Dataset size: {len(dataset)}")
        
        # Load one sample
        img, label = dataset[0]
        print(f"  Sample image shape: {img.shape}")
        print(f"  Sample label: {label}")
        print(f"  Image value range: [{img.min().item():.3f}, {img.max().item():.3f}]")
        
        print("✓ Dataset test passed!")
        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False


def test_config():
    """Test that the config file can be loaded."""
    print("\nTesting config file...")
    
    try:
        with open('config_vae_sthv2.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"  Model config: {config['model_params']}")
        print(f"  Data path: {config['data_params']['data_path']}")
        print(f"  Learning rate: {config['exp_params']['LR']}")
        print(f"  Batch size: {config['data_params']['train_batch_size']}")
        
        print("✓ Config test passed!")
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA availability...")
    
    if torch.cuda.is_available():
        print(f"  CUDA available: Yes")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Test model on GPU
        model = VanillaVAE(in_channels=3, latent_dim=256, hidden_dims=[64, 128, 256, 512, 1024], image_size=256)
        model = model.cuda()
        
        dummy_input = torch.randn(2, 3, 256, 256).cuda()
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f"  GPU forward pass successful!")
        print("✓ CUDA test passed!")
        return True
    else:
        print(f"  CUDA available: No")
        print("✗ CUDA test failed - no GPUs available")
        return False


def main():
    print("=" * 50)
    print("VAE Training Setup Test")
    print("=" * 50)
    
    results = {}
    
    # Test config
    results['config'] = test_config()
    
    # Test model
    results['model'] = test_model()
    
    # Test CUDA
    results['cuda'] = test_cuda()
    
    # Test dataset (only if config loaded successfully)
    if results['config']:
        with open('config_vae_sthv2.yaml', 'r') as f:
            config = yaml.safe_load(f)
        results['dataset'] = test_dataset(config['data_params']['data_path'])
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name.capitalize()}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 50)
    if all_passed:
        print("All tests passed! Ready to train.")
        print("\nTo start training:")
        print("  Multi-GPU: python train_ddp.py --config config_vae_sthv2.yaml")
        print("  Single-GPU: python train_single_gpu.py --config config_vae_sthv2.yaml")
    else:
        print("Some tests failed. Please fix the issues before training.")
    print("=" * 50)


if __name__ == '__main__':
    main()

