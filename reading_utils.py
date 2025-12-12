"""
Custom additions to reading_utils.py for stress prediction
Add these functions to the existing reading_utils.py file in DeepMind GNS code
"""

import tensorflow as tf
import functools


def parse_custom_timestep_example(example_proto, metadata):
    """
    Parse a single timestep Example from custom NPZ converter.
    
    This parser handles TFRecords created by npz_to_tfrecord_converter.py,
    which includes full particle data (positions, velocities, accelerations, stress).
    
    Args:
        example_proto: Serialized tf.train.Example
        metadata: Metadata dict with normalization statistics
    
    Returns:
        Dictionary with parsed and normalized features
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
        'n_particles_sediment': tf.io.FixedLenFeature([], tf.int64),
        'n_particles_boundary': tf.io.FixedLenFeature([], tf.int64),
        'time': tf.io.FixedLenFeature([], tf.float32),
        'timestep': tf.io.FixedLenFeature([], tf.int64),
    }
    
    # Parse the example
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    # Get particle count
    n_particles = parsed_features['n_particles_total']
    
    # Decode binary strings to arrays
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
    
    # Apply normalization using metadata statistics
    if 'vel_mean' in metadata:
        vel_mean = tf.constant(metadata['vel_mean'], dtype=tf.float32)
        vel_std = tf.constant(metadata['vel_std'], dtype=tf.float32)
        # Avoid division by zero
        vel_std = tf.maximum(vel_std, 1e-8)
        velocities = (velocities - vel_mean) / vel_std
    
    if 'acc_mean' in metadata:
        acc_mean = tf.constant(metadata['acc_mean'], dtype=tf.float32)
        acc_std = tf.constant(metadata['acc_std'], dtype=tf.float32)
        acc_std = tf.maximum(acc_std, 1e-8)
        accelerations = (accelerations - acc_mean) / acc_std
    
    if 'stress_mean' in metadata:
        stress_mean = tf.constant(metadata['stress_mean'], dtype=tf.float32)
        stress_std = tf.constant(metadata['stress_std'], dtype=tf.float32)
        stress_std = tf.maximum(stress_std, 1e-8)
        stress = (stress - stress_mean) / stress_std
    
    # Return in GNS-compatible format
    # We need to expand position to have a sequence dimension for compatibility
    # with existing GNS code that expects [seq_len, n_particles, 3]
    # For single timestep, seq_len = 1
    positions_seq = tf.expand_dims(positions, axis=0)  # [1, n_particles, 3]
    
    return {
        'position': positions_seq,
        'velocity': velocities,
        'particle_type': particle_types,
        'mass': mass,
        'density': density,
        'target_acceleration': accelerations,
        'target_stress': stress,
        'n_particles_per_example': n_particles,
    }


def load_custom_dataset(path, split, metadata, batch_size=1):
    """
    Load custom TFRecord dataset from npz_to_tfrecord_converter.py output.
    
    Args:
        path: Base path to data directory
        split: 'train', 'valid', or 'test'
        metadata: Metadata dictionary with normalization statistics
        batch_size: Number of examples per batch
    
    Returns:
        tf.data.Dataset batched and ready for training
    """
    import os
    tfrecord_path = os.path.join(path, f'{split}.tfrecord')
    
    if not tf.io.gfile.exists(tfrecord_path):
        raise ValueError(f"TFRecord file not found: {tfrecord_path}")
    
    # Create dataset from TFRecord
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # Parse examples
    parse_fn = functools.partial(parse_custom_timestep_example, metadata=metadata)
    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle for training
    if split == 'train':
        dataset = dataset.shuffle(buffer_size=1000)
    
    # Batch
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def compute_custom_dataset_statistics(tfrecord_path, max_examples=None):
    """
    Compute normalization statistics from custom TFRecord dataset.
    
    Use this to generate statistics for metadata.json.
    
    Args:
        tfrecord_path: Path to TFRecord file
        max_examples: Optional limit on number of examples to process
    
    Returns:
        Dictionary with mean and std for each feature
    """
    import numpy as np
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # Collect all data
    all_velocities = []
    all_accelerations = []
    all_stress = []
    
    for i, example_bytes in enumerate(dataset):
        if max_examples and i >= max_examples:
            break
        
        # Parse without normalization
        parsed = parse_custom_timestep_example(example_bytes, metadata={})
        
        # Extract numpy arrays
        vel = parsed['velocity'].numpy()
        acc = parsed['target_acceleration'].numpy()
        stress = parsed['target_stress'].numpy()
        
        all_velocities.append(vel)
        all_accelerations.append(acc)
        all_stress.append(stress)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} examples...")
    
    # Stack and compute statistics
    all_velocities = np.vstack(all_velocities)
    all_accelerations = np.vstack(all_accelerations)
    all_stress = np.vstack(all_stress)
    
    stats = {
        'vel_mean': all_velocities.mean(axis=0).tolist(),
        'vel_std': all_velocities.std(axis=0).tolist(),
        'acc_mean': all_accelerations.mean(axis=0).tolist(),
        'acc_std': all_accelerations.std(axis=0).tolist(),
        'stress_mean': all_stress.mean(axis=0).tolist(),
        'stress_std': all_stress.std(axis=0).tolist(),
    }
    
    print("\nComputed statistics:")
    print(f"Velocity mean: {stats['vel_mean']}")
    print(f"Velocity std: {stats['vel_std']}")
    print(f"Acceleration mean: {stats['acc_mean']}")
    print(f"Acceleration std: {stats['acc_std']}")
    print(f"Stress mean (first 3): {stats['stress_mean'][:3]}")
    print(f"Stress std (first 3): {stats['stress_std'][:3]}")
    
    return stats


# Example usage:
if __name__ == "__main__":
    # Compute statistics for metadata.json
    import json
    
    stats = compute_custom_dataset_statistics(
        'path/to/train.tfrecord',
        max_examples=1000  # Use subset for speed
    )
    
    # Load existing metadata
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Update with computed statistics
    metadata.update(stats)
    
    # Save updated metadata
    with open('metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nUpdated metadata.json with statistics")
