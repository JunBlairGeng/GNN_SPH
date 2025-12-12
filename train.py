"""
Example Training Script for GNN-SPH with Stress Prediction
This shows how to integrate all the custom modifications together.

This script is a simplified version of train.py with the custom modifications.
"""

import os
import json
import tensorflow as tf
from absl import app
from absl import flags

# Import custom reading utils (add to reading_utils.py)
from learning_to_simulate.reading_utils import (
    load_custom_dataset,
    compute_custom_dataset_statistics
)

# Import modified LearnedSimulator (modify learned_simulator.py)
from learning_to_simulate.learned_simulator import LearnedSimulator


# ============================================================================
# Command-line flags
# ============================================================================

FLAGS = flags.FLAGS

flags.DEFINE_string('data_path', None, 'Path to data directory')
flags.DEFINE_string('model_path', 'models/gnn_sph_stress', 'Path to save models')
flags.DEFINE_integer('batch_size', 2, 'Batch size for training')
flags.DEFINE_integer('num_steps', 10000000, 'Number of training steps')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
flags.DEFINE_integer('log_steps', 100, 'Steps between logging')
flags.DEFINE_integer('save_steps', 1000, 'Steps between checkpoints')
flags.DEFINE_bool('use_stress_prediction', True, 'Use dual decoder for stress')


def load_metadata(data_path):
    """Load or create metadata.json."""
    metadata_path = os.path.join(data_path, 'metadata.json')
    
    if os.path.exists(metadata_path):
        print(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        print(f"Metadata not found at {metadata_path}")
        print("Computing statistics from training data...")
        
        # Compute statistics from training data
        train_tfrecord = os.path.join(data_path, 'train.tfrecord')
        stats = compute_custom_dataset_statistics(
            train_tfrecord,
            max_examples=1000
        )
        
        # Create basic metadata
        metadata = {
            'bounds': [[0.0, 72.0], [0.0, 18.0], [0.0, 1.0]],
            'dim': 3,
            'dt': 0.0005,
            'sequence_length': 1,  # Single timestep
            'default_connectivity_radius': 1.5,
        }
        metadata.update(stats)
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")
    
    return metadata


def build_model(metadata, use_stress_prediction=True):
    """Build the LearnedSimulator model."""
    
    # Model hyperparameters
    num_dimensions = metadata['dim']
    connectivity_radius = metadata['default_connectivity_radius']
    
    # Graph network parameters
    graph_network_kwargs = {
        'latent_size': 128,
        'mlp_hidden_size': 128,
        'mlp_num_hidden_layers': 2,
        'num_message_passing_steps': 10,
    }
    
    # Particle types (0=boundary, 1=sediment)
    num_particle_types = 2
    particle_type_embedding_size = 16
    
    # Boundaries (from metadata)
    boundaries = metadata['bounds']
    
    # Normalization stats
    normalization_stats = {
        'acceleration': {
            'mean': metadata.get('acc_mean', [0.0] * num_dimensions),
            'std': metadata.get('acc_std', [1.0] * num_dimensions),
        },
        'velocity': {
            'mean': metadata.get('vel_mean', [0.0] * num_dimensions),
            'std': metadata.get('vel_std', [1.0] * num_dimensions),
        },
    }
    
    # Add stress normalization if using stress prediction
    if use_stress_prediction:
        normalization_stats['stress'] = {
            'mean': metadata.get('stress_mean', [0.0] * 9),
            'std': metadata.get('stress_std', [1.0] * 9),
        }
    
    # Build model
    model = LearnedSimulator(
        num_dimensions=num_dimensions,
        connectivity_radius=connectivity_radius,
        graph_network_kwargs=graph_network_kwargs,
        boundaries=boundaries,
        normalization_stats=normalization_stats,
        num_particle_types=num_particle_types,
        particle_type_embedding_size=particle_type_embedding_size,
        use_stress_prediction=use_stress_prediction,  # Custom flag
    )
    
    return model


def train(model, train_dataset, valid_dataset, metadata):
    """Training loop."""
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    
    # Checkpoint manager
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=FLAGS.model_path,
        max_to_keep=5
    )
    
    # Restore from checkpoint if exists
    latest_checkpoint = checkpoint_manager.latest_checkpoint
    if latest_checkpoint:
        print(f"Restoring from checkpoint: {latest_checkpoint}")
        checkpoint.restore(latest_checkpoint)
        start_step = int(checkpoint.save_counter)
    else:
        start_step = 0
    
    # TensorBoard writer
    train_writer = tf.summary.create_file_writer(
        os.path.join(FLAGS.model_path, 'train')
    )
    valid_writer = tf.summary.create_file_writer(
        os.path.join(FLAGS.model_path, 'valid')
    )
    
    # Training loop
    print(f"\nStarting training from step {start_step}...")
    print(f"Total steps: {FLAGS.num_steps}")
    print(f"Batch size: {FLAGS.batch_size}")
    print(f"Using stress prediction: {FLAGS.use_stress_prediction}\n")
    
    for step, batch in enumerate(train_dataset, start=start_step):
        if step >= FLAGS.num_steps:
            break
        
        # Training step
        with tf.GradientTape() as tape:
            if FLAGS.use_stress_prediction:
                # Use custom loss with stress
                loss_dict = model._loss_with_stress(batch, is_training=True)
                loss = loss_dict['loss']
            else:
                # Use standard GNS loss (acceleration only)
                loss = model.loss(batch, is_training=True)
        
        # Compute gradients and update
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Logging
        if step % FLAGS.log_steps == 0:
            if FLAGS.use_stress_prediction:
                print(f"Step {step}: "
                      f"loss={loss_dict['loss']:.4f}, "
                      f"acc_loss={loss_dict['acceleration_loss']:.4f}, "
                      f"stress_loss={loss_dict['stress_loss']:.4f}, "
                      f"physics_loss={loss_dict['physics_loss']:.4f}")
                
                # TensorBoard logging
                with train_writer.as_default():
                    tf.summary.scalar('loss/total', loss_dict['loss'], step=step)
                    tf.summary.scalar('loss/acceleration', loss_dict['acceleration_loss'], step=step)
                    tf.summary.scalar('loss/stress', loss_dict['stress_loss'], step=step)
                    tf.summary.scalar('loss/physics', loss_dict['physics_loss'], step=step)
            else:
                print(f"Step {step}: loss={loss:.4f}")
                with train_writer.as_default():
                    tf.summary.scalar('loss/total', loss, step=step)
        
        # Validation
        if step % (FLAGS.log_steps * 10) == 0 and step > 0:
            print(f"\nRunning validation at step {step}...")
            val_losses = []
            
            for val_batch in valid_dataset.take(10):
                if FLAGS.use_stress_prediction:
                    val_loss_dict = model._loss_with_stress(val_batch, is_training=False)
                    val_loss = val_loss_dict['loss']
                else:
                    val_loss = model.loss(val_batch, is_training=False)
                val_losses.append(val_loss.numpy())
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"Validation loss: {avg_val_loss:.4f}\n")
            
            with valid_writer.as_default():
                tf.summary.scalar('loss/total', avg_val_loss, step=step)
        
        # Save checkpoint
        if step % FLAGS.save_steps == 0 and step > 0:
            save_path = checkpoint_manager.save()
            print(f"Saved checkpoint at step {step}: {save_path}")
    
    print("\nTraining complete!")
    final_checkpoint = checkpoint_manager.save()
    print(f"Final checkpoint saved: {final_checkpoint}")


def main(_):
    """Main training function."""
    
    # Verify data path
    if not FLAGS.data_path:
        raise ValueError("Must specify --data_path")
    
    if not os.path.exists(FLAGS.data_path):
        raise ValueError(f"Data path does not exist: {FLAGS.data_path}")
    
    # Load metadata
    metadata = load_metadata(FLAGS.data_path)
    
    print("\nMetadata loaded:")
    print(f"  Bounds: {metadata['bounds']}")
    print(f"  Dimensions: {metadata['dim']}")
    print(f"  Timestep: {metadata['dt']}")
    print(f"  Connectivity radius: {metadata['default_connectivity_radius']}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = load_custom_dataset(
        FLAGS.data_path,
        'train',
        metadata,
        batch_size=FLAGS.batch_size
    )
    
    valid_dataset = load_custom_dataset(
        FLAGS.data_path,
        'valid',
        metadata,
        batch_size=FLAGS.batch_size
    )
    
    print("Datasets loaded successfully!")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(metadata, use_stress_prediction=FLAGS.use_stress_prediction)
    print("Model built successfully!")
    
    # Train
    train(model, train_dataset, valid_dataset, metadata)


if __name__ == '__main__':
    app.run(main)


# ============================================================================
# Usage Instructions
# ============================================================================

"""
To run this training script:

1. Prepare your data using npz_to_tfrecord_converter.py:
   
   python npz_to_tfrecord_converter.py \
       --input /path/to/npz_files \
       --output data/train.tfrecord
   
   (Repeat for valid.tfrecord and test.tfrecord)

2. Compute statistics and create metadata.json:
   
   The script will automatically compute statistics if metadata.json
   doesn't exist, or you can create it manually with:
   
   {
     "bounds": [[0.0, 72.0], [0.0, 18.0], [0.0, 1.0]],
     "dim": 3,
     "dt": 0.0005,
     "sequence_length": 1,
     "default_connectivity_radius": 1.5,
     "vel_mean": [0.0, 0.0, 0.0],
     "vel_std": [0.1, 0.1, 0.01],
     "acc_mean": [0.0, -9.81, 0.0],
     "acc_std": [5.0, 5.0, 1.0],
     "stress_mean": [0.0, ...],  # 9 values
     "stress_std": [1000.0, ...]  # 9 values
   }

3. Run training:
   
   # With stress prediction (dual decoder)
   python train_custom.py \
       --data_path=data \
       --model_path=models/gnn_sph_stress \
       --batch_size=2 \
       --use_stress_prediction=True
   
   # Without stress (baseline GNS)
   python train_custom.py \
       --data_path=data \
       --model_path=models/gnn_sph_baseline \
       --batch_size=2 \
       --use_stress_prediction=False

4. Monitor training with TensorBoard:
   
   tensorboard --logdir=models/gnn_sph_stress

5. For evaluation, see evaluate_custom.py (separate script)
"""
