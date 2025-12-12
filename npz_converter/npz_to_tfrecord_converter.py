#!/usr/bin/env python3
"""
NPZ to TFRecord Converter for GNN-SPH
This script converts PySPH NPZ simulation files to TFRecord format for GNN training.
"""

import os
import sys
import glob
import numpy as np
import tensorflow as tf
from utils import load as pysph_load


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def extract_particle_data(particle_array, particle_type):
    """
    Extract relevant data from PySPH ParticleArray for GNN-SPH.
    
    Args:
        particle_array: PySPH ParticleArray object
        particle_type: int, particle type (0 for boundary, 1 for sediment/soil)
    
    Returns:
        dict: Dictionary with extracted arrays
    """
    n_particles = particle_array.get_number_of_particles()
    
    # Get position
    x, y, z = particle_array.get('x', 'y', 'z')
    positions = np.stack([x, y, z], axis=1).astype(np.float32)
    
    # Get velocity
    u, v, w = particle_array.get('u', 'v', 'w')
    velocities = np.stack([u, v, w], axis=1).astype(np.float32)
    
    # Get acceleration
    au, av, aw = particle_array.get('au', 'av', 'aw')
    accelerations = np.stack([au, av, aw], axis=1).astype(np.float32)
    
    # Get stress tensor (strided with stride 9 for 3x3 matrix)
    sigma = particle_array.get('sigma')
    # Reshape from strided format to (n_particles, 9)
    stress = sigma.reshape(n_particles, 9).astype(np.float32)
    
    # Get particle types
    particle_types = np.full(n_particles, particle_type, dtype=np.int32)
    
    # Get other useful properties
    try:
        mass = particle_array.get('m')
        rho = particle_array.get('rho')
    except:
        mass = np.ones(n_particles, dtype=np.float32)
        rho = np.ones(n_particles, dtype=np.float32)
    
    return {
        'positions': positions,
        'velocities': velocities,
        'accelerations': accelerations,
        'stress': stress,
        'particle_types': particle_types,
        'mass': mass.astype(np.float32),
        'density': rho.astype(np.float32),
        'n_particles': n_particles
    }


def npz_to_tfrecord_example(npz_file):
    """
    Convert a single NPZ file to a TFRecord example.
    
    Args:
        npz_file: Path to NPZ file
    
    Returns:
        tf.train.Example: TFRecord example
    """
    # Load data using pysph_load
    data = pysph_load(npz_file)
    
    particle_arrays = data['arrays']
    solver_data = data['solver_data']
    
    # Extract sediment (soil) particles - type 1
    sediment_data = extract_particle_data(particle_arrays['sediment'], particle_type=1)
    
    # Extract boundary particles - type 0
    boundary_data = extract_particle_data(particle_arrays['boundary'], particle_type=0)
    
    # Combine all particles
    positions = np.vstack([sediment_data['positions'], boundary_data['positions']])
    velocities = np.vstack([sediment_data['velocities'], boundary_data['velocities']])
    accelerations = np.vstack([sediment_data['accelerations'], boundary_data['accelerations']])
    stress = np.vstack([sediment_data['stress'], boundary_data['stress']])
    particle_types = np.concatenate([sediment_data['particle_types'], boundary_data['particle_types']])
    mass = np.concatenate([sediment_data['mass'], boundary_data['mass']])
    density = np.concatenate([sediment_data['density'], boundary_data['density']])
    
    # Create feature dictionary
    feature = {
        'positions': _bytes_feature(positions.tobytes()),
        'velocities': _bytes_feature(velocities.tobytes()),
        'accelerations': _bytes_feature(accelerations.tobytes()),
        'stress': _bytes_feature(stress.tobytes()),
        'particle_types': _bytes_feature(particle_types.tobytes()),
        'mass': _bytes_feature(mass.tobytes()),
        'density': _bytes_feature(density.tobytes()),
        'n_particles_sediment': _int64_feature(sediment_data['n_particles']),
        'n_particles_boundary': _int64_feature(boundary_data['n_particles']),
        'n_particles_total': _int64_feature(len(particle_types)),
        'time': _float_feature(float(solver_data['t'])),
        'timestep': _float_feature(float(solver_data['dt'])),
    }
    
    # Create example
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    return example


def convert_npz_sequence_to_tfrecord(input_dir, output_file, pattern='*.npz'):
    """
    Convert a sequence of NPZ files to a single TFRecord file.
    
    Args:
        input_dir: Directory containing NPZ files
        output_file: Output TFRecord file path
        pattern: Glob pattern for NPZ files (default: '*.npz')
    """
    # Get all NPZ files
    npz_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    
    if not npz_files:
        raise ValueError(f"No NPZ files found in {input_dir} with pattern {pattern}")
    
    print(f"Found {len(npz_files)} NPZ files")
    
    # Write to TFRecord
    with tf.io.TFRecordWriter(output_file) as writer:
        for i, npz_file in enumerate(npz_files):
            try:
                print(f"Processing {i+1}/{len(npz_files)}: {os.path.basename(npz_file)}")
                example = npz_to_tfrecord_example(npz_file)
                writer.write(example.SerializeToString())
            except Exception as e:
                print(f"Error processing {npz_file}: {e}")
                continue
    
    print(f"\nSuccessfully created TFRecord: {output_file}")


def convert_single_npz_to_tfrecord(npz_file, output_file):
    """
    Convert a single NPZ file to a TFRecord file (for testing).
    
    Args:
        npz_file: Path to NPZ file
        output_file: Output TFRecord file path
    """
    print(f"Converting {npz_file} to {output_file}")
    
    with tf.io.TFRecordWriter(output_file) as writer:
        example = npz_to_tfrecord_example(npz_file)
        writer.write(example.SerializeToString())
    
    print(f"Successfully created TFRecord: {output_file}")


def read_tfrecord_example(tfrecord_file, max_examples=1):
    """
    Read and print information from a TFRecord file (for verification).
    
    Args:
        tfrecord_file: Path to TFRecord file
        max_examples: Maximum number of examples to read
    """
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    
    for i, serialized_example in enumerate(dataset.take(max_examples)):
        example = tf.train.Example()
        example.ParseFromString(serialized_example.numpy())
        
        features = example.features.feature
        
        print(f"\n=== Example {i+1} ===")
        print(f"Time: {features['time'].float_list.value[0]:.6f}")
        print(f"Timestep: {features['timestep'].float_list.value[0]:.6f}")
        print(f"Total particles: {features['n_particles_total'].int64_list.value[0]}")
        print(f"Sediment particles: {features['n_particles_sediment'].int64_list.value[0]}")
        print(f"Boundary particles: {features['n_particles_boundary'].int64_list.value[0]}")
        
        # Decode positions to verify
        n_total = features['n_particles_total'].int64_list.value[0]
        positions_bytes = features['positions'].bytes_list.value[0]
        positions = np.frombuffer(positions_bytes, dtype=np.float32).reshape(n_total, 3)
        print(f"Positions shape: {positions.shape}")
        print(f"Position range: x=[{positions[:,0].min():.2f}, {positions[:,0].max():.2f}], "
              f"y=[{positions[:,1].min():.2f}, {positions[:,1].max():.2f}], "
              f"z=[{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert NPZ files to TFRecord format for GNN-SPH')
    parser.add_argument('--input', '-i', required=True, help='Input NPZ file or directory')
    parser.add_argument('--output', '-o', required=True, help='Output TFRecord file')
    parser.add_argument('--pattern', '-p', default='*.npz', help='Pattern for NPZ files (default: *.npz)')
    parser.add_argument('--verify', '-v', action='store_true', help='Verify TFRecord after creation')
    
    args = parser.parse_args()
    
    # Convert
    if os.path.isdir(args.input):
        convert_npz_sequence_to_tfrecord(args.input, args.output, args.pattern)
    else:
        convert_single_npz_to_tfrecord(args.input, args.output)
    
    # Verify
    if args.verify:
        print("\n" + "="*60)
        print("Verifying TFRecord file...")
        print("="*60)
        read_tfrecord_example(args.output, max_examples=2)
