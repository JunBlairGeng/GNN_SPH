import functools
import numpy as np
import tensorflow.compat.v1 as tf

# Create a description of the features.
_FEATURE_DESCRIPTION = {
    'position': tf.io.VarLenFeature(tf.string),
}

_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT['step_context'] = tf.io.VarLenFeature(
    tf.string)

_FEATURE_DTYPES = {
    'position': {
        'in': np.float32,
        'out': tf.float32
    },
    'step_context': {
        'in': np.float32,
        'out': tf.float32
    }
}

_CONTEXT_FEATURES = {
    'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'particle_type': tf.io.VarLenFeature(tf.string)
}


def convert_to_tensor(x, encoded_dtype):
  if len(x) == 1:
    out = np.frombuffer(x[0].numpy(), dtype=encoded_dtype)
  else:
    out = []
    for el in x:
      out.append(np.frombuffer(el.numpy(), dtype=encoded_dtype))
  out = tf.convert_to_tensor(np.array(out))
  return out


def parse_serialized_simulation_example(example_proto, metadata):
  """Parses a serialized simulation tf.SequenceExample.

  Args:
    example_proto: A string encoding of the tf.SequenceExample proto.
    metadata: A dict of metadata for the dataset.

  Returns:
    context: A dict, with features that do not vary over the trajectory.
    parsed_features: A dict of tf.Tensors representing the parsed examples
      across time, where axis zero is the time axis.

  """
  if 'context_mean' in metadata:
    feature_description = _FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT
  else:
    feature_description = _FEATURE_DESCRIPTION
  context, parsed_features = tf.io.parse_single_sequence_example(
      example_proto,
      context_features=_CONTEXT_FEATURES,
      sequence_features=feature_description)
  for feature_key, item in parsed_features.items():
    convert_fn = functools.partial(
        convert_to_tensor, encoded_dtype=_FEATURE_DTYPES[feature_key]['in'])
    parsed_features[feature_key] = tf.py_function(
        convert_fn, inp=[item.values], Tout=_FEATURE_DTYPES[feature_key]['out'])

  # There is an extra frame at the beginning so we can calculate pos change
  # for all frames used in the paper.
  position_shape = [metadata['sequence_length'] + 1, -1, metadata['dim']]

  # Reshape positions to correct dim:
  parsed_features['position'] = tf.reshape(parsed_features['position'],
                                           position_shape)
  # Set correct shapes of the remaining tensors.
  sequence_length = metadata['sequence_length'] + 1
  if 'context_mean' in metadata:
    context_feat_len = len(metadata['context_mean'])
    parsed_features['step_context'] = tf.reshape(
        parsed_features['step_context'],
        [sequence_length, context_feat_len])
  # Decode particle type explicitly
  context['particle_type'] = tf.py_function(
      functools.partial(convert_fn, encoded_dtype=np.int64),
      inp=[context['particle_type'].values],
      Tout=[tf.int64])
  context['particle_type'] = tf.reshape(context['particle_type'], [-1])
  return context, parsed_features


def split_trajectory(context, features, window_length=7):
  """Splits trajectory into sliding windows."""
  # Our strategy is to make sure all the leading dimensions are the same size,
  # then we can use from_tensor_slices.

  trajectory_length = features['position'].get_shape().as_list()[0]

  # We then stack window_length position changes so the final
  # trajectory length will be - window_length +1 (the 1 to make sure we get
  # the last split).
  input_trajectory_length = trajectory_length - window_length + 1

  model_input_features = {}
  # Prepare the context features per step.
  model_input_features['particle_type'] = tf.tile(
      tf.expand_dims(context['particle_type'], axis=0),
      [input_trajectory_length, 1])

  if 'step_context' in features:
    global_stack = []
    for idx in range(input_trajectory_length):
      global_stack.append(features['step_context'][idx:idx + window_length])
    model_input_features['step_context'] = tf.stack(global_stack)

  pos_stack = []
  for idx in range(input_trajectory_length):
    pos_stack.append(features['position'][idx:idx + window_length])
  # Get the corresponding positions
  model_input_features['position'] = tf.stack(pos_stack)

  return tf.data.Dataset.from_tensor_slices(model_input_features)


def parse_custom_timestep_example(example_proto, metadata):
    """
    Parse a single timestep Example from custom NPZ converter.
    
    Args:
        example_proto: Serialized tf.train.Example
        metadata: Metadata dict (same as GNS metadata.json)
    
    Returns:
        Dictionary with parsed features
    """
    # Define feature description
    feature_description = {
        # Inputs
        'positions': tf.io.FixedLenFeature([], tf.string),
        'velocities': tf.io.FixedLenFeature([], tf.string),
        'particle_types': tf.io.FixedLenFeature([], tf.string),
        'mass': tf.io.FixedLenFeature([], tf.string),
        'density': tf.io.FixedLenFeature([], tf.string),
        
        # Targets
        'accelerations': tf.io.FixedLenFeature([], tf.string),
        'stress': tf.io.FixedLenFeature([], tf.string),
        
        # Metadata
        'n_particles_total': tf.io.FixedLenFeature([], tf.int64),
        'timestep': tf.io.FixedLenFeature([], tf.int64),
    }
    
    # Parse the example
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    # Decode binary strings to arrays
    n_particles = parsed_features['n_particles_total']
    
    positions = tf.io.decode_raw(parsed_features['positions'], tf.float32)
    positions = tf.reshape(positions, [n_particles, 3])
    
    velocities = tf.io.decode_raw(parsed_features['velocities'], tf.float32)
    velocities = tf.reshape(velocities, [n_particles, 3])
    
    particle_types = tf.io.decode_raw(parsed_features['particle_types'], tf.int32)
    particle_types = tf.reshape(particle_types, [n_particles])
    
    mass = tf.io.decode_raw(parsed_features['mass'], tf.float32)
    mass = tf.reshape(mass, [n_particles])
    
    density = tf.io.decode_raw(parsed_features['density'], tf.float32)
    density = tf.reshape(density, [n_particles])
    
    # Targets
    accelerations = tf.io.decode_raw(parsed_features['accelerations'], tf.float32)
    accelerations = tf.reshape(accelerations, [n_particles, 3])
    
    stress = tf.io.decode_raw(parsed_features['stress'], tf.float32)
    stress = tf.reshape(stress, [n_particles, 9])  # 3x3 = 9 components
    
    # Normalize using metadata statistics
    if 'vel_mean' in metadata:
        vel_mean = tf.constant(metadata['vel_mean'], dtype=tf.float32)
        vel_std = tf.constant(metadata['vel_std'], dtype=tf.float32)
        velocities = (velocities - vel_mean) / vel_std
    
    if 'acc_mean' in metadata:
        acc_mean = tf.constant(metadata['acc_mean'], dtype=tf.float32)
        acc_std = tf.constant(metadata['acc_std'], dtype=tf.float32)
        accelerations = (accelerations - acc_mean) / acc_std
    
    # Return in GNS-compatible format
    return {
        'position': positions,
        'velocity': velocities,
        'particle_type': particle_types,
        'mass': mass,
        'density': density,
        'target_acceleration': accelerations,
        'target_stress': stress,
        'n_particles_per_example': n_particles,
    }


def load_custom_dataset(path, split, metadata):
    """
    Load custom TFRecord dataset.
    
    Args:
        path: Base path to data directory
        split: 'train', 'valid', or 'test'
        metadata: Metadata dictionary
    
    Returns:
        tf.data.Dataset
    """
    import os
    tfrecord_path = os.path.join(path, f'{split}.tfrecord')
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(
        lambda x: parse_custom_timestep_example(x, metadata),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    return dataset
