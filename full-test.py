import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['JAX_PLATFORMS'] = 'tpu,cpu'
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Be quiet

import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import jax_utils
import optax
from flax.training import train_state
import time
from jax.sharding import PositionalSharding
import json

from Preprocess import prepare_tcn_data
from TCN_model import TCN, init_train_state, train_step

def load_config(config_path="full-test-config.txt"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"Configuration loaded: {config['test_name']}")
        return config
    except Exception as e:
        print(f"Fail to load configuration: {e}")
        return None

config = load_config()

# Load Configuration
if config:
    BATCH_SIZES = config['batch_size_list']
    TIME_STEP = config['time_step']
    BUFFER_SIZE = config['buffer_size']
    LR = config['learning_rate']
    TOTAL_STEPS = config['total_steps_per_test']
    
    print(f"Testing Batch sizes: {BATCH_SIZES}")

# Load Data
file_location = "gs://fi2010-lob-data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_9.txt"
with tf.io.gfile.GFile(file_location, 'r') as f:
    raw_data = np.loadtxt(f)

# JAX Sharding
num_devices = jax.local_device_count() 
sharding = PositionalSharding(jax.devices())

x_sharding = sharding.reshape(num_devices, 1, 1)
y_sharding = sharding.reshape(num_devices)

# define function for single batch size
def test_for_batch(BATCH_SIZE):
    # Data Preprocessing
    train_dataset = prepare_tcn_data(raw_data, BATCH_SIZE, TIME_STEP, True, True, BUFFER_SIZE)
    train_iter = train_dataset.as_numpy_iterator()

    # random seed
    rng = jax.random.PRNGKey(3721)
    rng, dropout_rng = jax.random.split(rng)
    # Create Model and training state
    model = TCN()
    state, dropout_rng = init_train_state(rng, model)

    # Data warmup
    next_batch = next(train_iter)
    parallel_x = next_batch[0].astype(jnp.bfloat16)
    parallel_y = next_batch[1].astype(jnp.int32)
    parallel_x = jax.device_put(parallel_x, x_sharding)
    parallel_y = jax.device_put(parallel_y, y_sharding)

    start_time = time.time()
    running_loss = 0.0
    XLA_time = 0.0

    print(f"--- Testing Batch Size: {BATCH_SIZE}. ---")
    for step in range(TOTAL_STEPS):
        # Calculation
        state, loss, acc = train_step(state, parallel_x, parallel_y, dropout_rng)
        
        # Asynchronous extraction
        next_batch = next(train_iter)
        cpu_x = next_batch[0].astype(jnp.bfloat16)
        cpu_y = next_batch[1].astype(jnp.int32)
        
        parallel_x = jax.device_put(cpu_x, x_sharding)
        parallel_y = jax.device_put(cpu_y, y_sharding)
        
        # Returning Single Number
        running_loss += loss.item()

        if step == 0:
            XLA_time = time.time() - start_time
        if step == 1:
            XLA_time_2 = time.time() - start_time - XLA_time
        if step % 50 == 0:
            print(f" Step {step:04d} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}")

    total_time = time.time() - start_time
    print(f"✅ Done {TOTAL_STEPS} Steps! Total time: {total_time:.2f}s")
    print(f'🚀 XLA Time + first step(approximate): {XLA_time}.')
    print(f'🚀 Second step: {XLA_time_2}.')
    sec_per_step = (total_time-XLA_time-XLA_time_2)/(TOTAL_STEPS - 2)
    print(f'⚡ Second per step: {sec_per_step:.8f}s ')
    samples_per_sec = round(BATCH_SIZE*(1/sec_per_step),0)
    print(f"📊 Data I/O: {samples_per_sec} samples/sec")
    
    return {
        'Batch size': BATCH_SIZE,
        'Total wall time': total_time,
        'First step': XLA_time,
        'Second step': XLA_time_2,
        'Second per step': sec_per_step,
        'Samples per second': samples_per_sec
    }

result_all = []

for BATCH_SIZE in BATCH_SIZES:
    result = test_for_batch(BATCH_SIZE)
    result_all.append(result)

with open("Test_Results.txt", "w", encoding="utf-8") as f:
    json.dump(result_all, f, indent=4)

print("✅ All results have been saved to Phase2_Test_Results.txt")