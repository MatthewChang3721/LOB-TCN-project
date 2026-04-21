import jax
import jax.numpy as jnp
import flax.linen as nn
import flax
from flax.training import train_state
import optax
from typing import Sequence
import functools

class TCNBlock(nn.Module):
    features: int
    dilation: int
    kernel_size: int = 3
    dropout_rate: float = 0.2

    @nn.compact 
    def __call__(self, x, train: bool = True):
        # residual learning
        res = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            kernel_dilation=(self.dilation,),
            padding='CAUSAL',
            dtype = jnp.bfloat16
        )(x)
        res = nn.relu(res)
        res = nn.Dropout(rate = self.dropout_rate)(res, deterministic= not train)

        # residual connection
        if x.shape[-1] != self.features:
            connection = nn.Conv(
                features = self.features,
                kernel_size = (1,),
                padding = 'SAME',
                dtype = jnp.bfloat16
            )(x)
        else:
            connection = x
        return nn.relu(res + connection)

class TCN(nn.Module):
    features: int = 64
    nn_dilation: Sequence[int] = (1, 2, 4, 8, 16, 32)
    num_classes: int = 3

    @nn.compact
    def __call__(self, x, train: bool = True):
        # transfer data type to bfloat16
        x = x.astype(jnp.bfloat16)
        
        for dilation in self.nn_dilation:
            x = TCNBlock(
                features=self.features, 
                dilation=dilation, 
                kernel_size=3,
                dropout_rate = 0.2,
            )(x, train = train)
            
        # Slice the last timestamp
        Output = x[:, -1, :]
        logits = nn.Dense(features=self.num_classes, dtype = jnp.float32)(Output)
        return logits

# initiate a train weight matrix for TCN
def init_train_state(rng, model, learning_rate = 0.001):
    params_rng, dropout_rng = jax.random.split(rng)
    
    dummy_x = jnp.ones((1, 127, 40)) 
    variables = model.init(
        {'params': params_rng,'dropout': dropout_rng},
        dummy_x,
        train = False
    )
    params = variables['params'] 

    tx = optax.adam(learning_rate)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    ), dropout_rng

@functools.partial(jax.jit, donate_argnums=(0,))
def train_step(state, batch_x, batch_y, dropout_rng):
    # loss function
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params}, 
            batch_x,
            train = True,
            rngs = {'dropout':dropout_rng}            
        )
        
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch_y
        ).mean()
        
        return loss, logits 
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    
    (loss, logits), grads = grad_fn(state.params)
    # calucation for multiple TPU:
    # grads = jax.lax.pmean(grads, axis_name='batch') 

    state = state.apply_gradients(grads=grads)

    pred = jnp.argmax(logits, axis= -1)
    accuracy = jnp.mean(pred == batch_y)
    
    return state, loss, accuracy

# this function is for calculation on multiple tpus. "batch": this is the sychronize columns between tpus
@functools.partial(jax.pmap, axis_name='batch')
def train_step_tpu(state, batch_x, batch_y, dropout_rng):
    # loss function remains the same
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params}, 
            batch_x,
            train = True,
            rngs = {'dropout':dropout_rng}            
        )
        
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch_y
        ).mean()
        
        return loss, logits
    
    # 1. Functional Transformation: Wrap the loss function with JAX's automatic differentiation engine.
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True) 
    # 2. Execution: Compute the actual values by passing the parameter matrix from the current state.
    (loss, logits), grads = grad_fn(state.params) 
    
    # Cross-Device Synchronization (All-Reduce)
    # Average the gradients across all 4 TPU cores before applying them.
    grads = jax.lax.pmean(grads, axis_name='batch')

    state = state.apply_gradients(grads=grads)

    # Calculate metrics
    pred = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(pred == batch_y)
    
    # Synchronize metrics across devices for accurate logging
    loss = jax.lax.pmean(loss, axis_name='batch')
    accuracy = jax.lax.pmean(accuracy, axis_name='batch')
    
    return state, loss, accuracy

if __name__ == "__main__":
    rng = jax.random.PRNGKey(42) 
    tcn_model_test = TCN()
    
    state, dropout_rng = init_train_state(rng, tcn_model_test, learning_rate=1e-3)
    dropout_rng = jax.random.split(dropout_rng, 4)

    state = flax.jax_utils.replicate(state)
    
    x_test = jnp.ones((4,4,127,40),dtype = jnp.bfloat16)
    y_test = jnp.zeros((4,4), dtype = jnp.int32)

    state, loss, accuracy = train_step_tpu(state, x_test, y_test, dropout_rng)
    print(f"Loss: {loss}, Accuracy: {accuracy}")