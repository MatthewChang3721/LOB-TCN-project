# LOB-TCN Project

This project implements a highly efficient TCN (Temporal Convolutional Network) model based on the JAX and Flax frameworks, specifically designed for Limit Order Book (LOB) data modeling and price trend prediction in high-frequency financial scenarios.

## 🚀 Key Features

- **High-Performance Backend**: Built with JAX/Flax, fully supporting XLA compilation acceleration.
- **Mixed Precision Training**: Core convolutional computations utilize the bfloat16 data type to optimize memory footprint and boost computational speed (tailored for TPUs and next-gen GPUs).
- **Robust Data Pipeline**: Leverages tf.data to build a highly efficient preprocessing pipeline, supporting sliding window sampling for large-scale data and multi-host/multi-device sharding.
- **Residual Architecture**: Implements TCN blocks with Causal Convolutions (Causal Conv) and Dilated Convolutions (Dilation Conv), perfectly suited for capturing long-term historical dependencies in time-series data.

## 📁 Project Structure

- `TCN_model.py`: Core model definition file. Contains the TCNBlock module, the main TCN architecture, parameter initialization functions, and the jax.jit-compiled training step function.
- `Preprocess.py`: Data preprocessing utilities. Contains the prepare_tcn_data function, which converts raw numpy arrays into the tf.data.Dataset format with built-in support for multi-host distributed sampling.
- `lob-tcn-demo.ipynb`: A getting-started notebook. Demonstrates the complete end-to-end workflow: loading the FI-2010 dataset from Google Cloud Storage (GCS), initializing the model, and running a basic training loop.
- `lob-tcn-test.ipynb`:  Modular testing script. Used to verify the integration and synergy between the Preprocess and TCN_model modules.
 
## 🛠️ Requirements

Ensure the following libraries are installed in your environment:

JAX & JAXlib
Flax (Linen)
TensorFlow (for the data pipeline)
Optax (optimizer)
NumPy

## 📖 Quick Start

1. **Data Preparation**: The project uses the FI-2010 LOB dataset by default. You can refer to lob-tcn-demo.ipynb for instructions on how to load it from Google Cloud Storage.
2. **Run Training**:
   ```python
   from TCN_model import TCN, init_train_state, train_step
   from Preprocess import prepare_tcn_data

   train_dataset = prepare_tcn_data(raw_data, batch_size, time_step, True, False)
   
   # Initialize model state
   rng = jax.random.PRNGKey(3721)
   rng, dropout_rng = jax.random.split(rng)
   model = TCN()
   state, dropout_rng = init_train_state(rng, model)
   
   # Execute a training step
   state, loss, acc = train_step(state, batch_x, batch_y, dropout_rng)
   ```


## 🧠  Model Details

The default TCN model configuration includes:
- **Feature Dimension**: 64
- **Dilation Sequence**: (1, 2, 4, 8, 16, 32)
- **Kernel Size**: 3
- **Dropout**: 0.2
- **Output Layer**: 3 classes (corresponding to price Up, Flat, Down)
