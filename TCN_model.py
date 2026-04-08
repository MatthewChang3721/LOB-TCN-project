import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Sequence

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

@jax.jit
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
    # TPU 优化：如果是多机多卡，这里需要对梯度进行 all-reduce
    # grads = jax.lax.pmean(grads, axis_name='batch') 

    state = state.apply_gradients(grads=grads)

    pred = jnp.argmax(logits, axis= -1)
    accuracy = jnp.mean(pred == batch_y)
    
    return state, loss, accuracy

if __name__ == "__main__":
    rng = jax.random.PRNGKey(42) 
    tcn_model_test = TCN()
    
    state = init_train_state(rng, tcn_model_test, learning_rate=1e-3)
    
    x_test = jnp.ones((1,127,40),dtype = jnp.bfloat16)
    y_test = jnp.zeros((1,), dtype = jnp.int32)

    rng, step_rng = jax.random.split(rng)
    state, loss, accuracy = train_step(state, x_test, y_test, step_rng)
    print(f"Loss: {loss}, Accuracy: {accuracy}")