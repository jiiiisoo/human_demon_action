"""
Suppress TensorFlow warnings - Import this FIRST before anything else
"""
import os
import sys
import warnings

# Must be set before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide all CUDA devices from TensorFlow

# Suppress Python warnings
warnings.filterwarnings('ignore')

# Redirect stderr temporarily while importing TensorFlow
import io
old_stderr = sys.stderr
sys.stderr = io.StringIO()

try:
    import tensorflow as tf
    # Configure TensorFlow
    tf.config.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
finally:
    # Restore stderr
    sys.stderr = old_stderr

# Suppress absl logging
try:
    import absl.logging
    absl.logging.set_verbosity('error')
    absl.logging.set_stderrthreshold('error')
except:
    pass

print("✅ TensorFlow warnings suppressed")

