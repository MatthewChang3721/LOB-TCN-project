import numpy as np
import tensorflow as tf
import jax

def prepare_tcn_data(raw_data, batch_size = 1024, time_step = 127, shuffle = True, repeat = False):
    # Extract features and label
    raw_data = raw_data.T
    features = raw_data[:,:40].astype(np.float32)
    label = (raw_data[:,-1] - 1).astype(np.int32)

    # Create Dataset (No Batching for better sharding)
    dataset = tf.keras.utils.timeseries_dataset_from_array(
            data = features,
            targets = label[time_step - 1:], 
            sequence_length = time_step,
            sequence_stride = 1,
            batch_size = None,
    )

    # Multi-host
    if jax.process_count() > 1:
        host_id = jax.process_index()
        dataset = dataset.shard(num_shards=jax.process_count(), index=host_id)

    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(buffer_size = 10000)

    # Repeat
    if repeat:
        dataset = dataset.repeat()
    
    # Batch (drop_remainder)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    # Prefech
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset