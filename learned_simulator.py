"""
Modifications to learned_simulator.py for dual decoder (acceleration + stress)

These modifications extend the DeepMind GNS LearnedSimulator class to:
1. Add a dual decoder architecture (acceleration + stress prediction)
2. Implement physics-based loss functions
3. Handle stress tensor prediction

Add these methods to the LearnedSimulator class.
"""

import tensorflow as tf
import sonnet as snt


# ============================================================================
# Add these methods to the LearnedSimulator class
# ============================================================================

def _build_dual_decoder(self):
    """
    Build dual decoder architecture for predicting acceleration and stress.
    
    Replace the single _output_network with two separate decoders.
    Call this in __init__ instead of building single output network.
    """
    # Helper function for building MLPs (use existing from learned_simulator.py)
    def get_mlp_fn(output_size, name=None):
        """Build MLP with layer norm."""
        return snt.Sequential([
            snt.nets.MLP(
                [128, 128, output_size],
                activate_final=False,
                activation=tf.nn.relu,
                name=name
            ),
            snt.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True,
                name=f'{name}_layer_norm' if name else 'layer_norm'
            )
        ])
    
    # Decoder 1: Acceleration (3D output)
    self._acceleration_decoder = get_mlp_fn(
        output_size=3,
        name='acceleration_decoder'
    )
    
    # Decoder 2: Stress tensor (9D output for 3x3 symmetric matrix)
    self._stress_decoder = get_mlp_fn(
        output_size=9,
        name='stress_decoder'
    )
    
    print("Built dual decoder architecture:")
    print("  - Acceleration decoder: output_size=3")
    print("  - Stress decoder: output_size=9")


def _predict_acceleration_and_stress(
    self,
    position_sequence,
    n_particles_per_example,
    particle_types,
    global_context=None
):
    """
    Predict both acceleration and stress from particle state.
    
    This is the forward pass through the dual decoder network.
    
    Args:
        position_sequence: Particle positions [batch, seq_len, n_particles, 3]
        n_particles_per_example: Number of particles per example
        particle_types: Particle type indices [batch, n_particles]
        global_context: Optional global features
    
    Returns:
        Tuple of (predicted_acceleration, predicted_stress)
        - predicted_acceleration: [batch, n_particles, 3]
        - predicted_stress: [batch, n_particles, 9]
    """
    # Encode particle features and build graph
    # (Use existing _encoder_preprocessor from base class)
    graph = self._encoder_preprocessor(
        position_sequence,
        n_particles_per_example,
        particle_types,
        global_context
    )
    
    # Process through graph network
    # (Use existing _processor from base class)
    for _ in range(self._num_processing_steps):
        graph = self._processor(graph)
    
    # Decode graph to per-node features
    # (Use existing _decoder_postprocessor from base class)
    per_node_features = self._decoder_postprocessor(
        graph,
        n_particles_per_example
    )
    
    # Split into two predictions using separate decoders
    predicted_acceleration = self._acceleration_decoder(per_node_features)
    predicted_stress = self._stress_decoder(per_node_features)
    
    return predicted_acceleration, predicted_stress


def _loss_with_stress(self, inputs, is_training=True):
    """
    Compute loss including both acceleration and stress prediction.
    
    This replaces the standard loss function with one that handles
    dual outputs and physics-based constraints.
    
    Args:
        inputs: Dictionary with:
            - 'position': [batch, seq_len, n_particles, 3]
            - 'particle_type': [batch, n_particles]
            - 'target_acceleration': [batch, n_particles, 3]
            - 'target_stress': [batch, n_particles, 9]
            - 'n_particles_per_example': scalar
        is_training: Whether in training mode (for dropout etc.)
    
    Returns:
        Dictionary with loss components:
            - 'loss': total weighted loss
            - 'acceleration_loss': MSE on acceleration
            - 'stress_loss': MSE on stress
            - 'physics_loss': physics-based constraints
    """
    # Get predictions from dual decoder
    predicted_acceleration, predicted_stress = self._predict_acceleration_and_stress(
        position_sequence=inputs['position'],
        n_particles_per_example=inputs['n_particles_per_example'],
        particle_types=inputs['particle_type'],
    )
    
    # Ground truth targets
    target_acceleration = inputs['target_acceleration']
    target_stress = inputs['target_stress']
    
    # Particle type mask (only compute loss on sediment particles, type=1)
    particle_types = inputs['particle_type']
    sediment_mask = tf.cast(tf.equal(particle_types, 1), tf.float32)
    sediment_mask_3d = tf.expand_dims(sediment_mask, -1)  # [batch, n_particles, 1]
    
    # 1. Acceleration loss (MSE on sediment particles only)
    acc_error = (predicted_acceleration - target_acceleration) ** 2
    acc_loss = tf.reduce_sum(acc_error * sediment_mask_3d) / tf.reduce_sum(sediment_mask_3d)
    
    # 2. Stress loss (MSE on sediment particles only)
    # Extend mask to 9 dimensions for stress tensor
    sediment_mask_9d = tf.expand_dims(sediment_mask, -1)
    sediment_mask_9d = tf.tile(sediment_mask_9d, [1, 1, 9])  # [batch, n_particles, 9]
    
    stress_error = (predicted_stress - target_stress) ** 2
    stress_loss = tf.reduce_sum(stress_error * sediment_mask_9d) / tf.reduce_sum(sediment_mask_9d)
    
    # 3. Physics-based constraints
    physics_loss = self._compute_physics_loss(
        predicted_acceleration=predicted_acceleration,
        predicted_stress=predicted_stress,
        inputs=inputs,
        sediment_mask=sediment_mask
    )
    
    # Weighted combination (tune these weights for your problem)
    # These weights emphasize acceleration prediction while regularizing stress
    total_loss = (
        1.0 * acc_loss +           # Acceleration loss (primary objective)
        0.5 * stress_loss +         # Stress loss (secondary objective)
        0.1 * physics_loss          # Physics constraints (regularization)
    )
    
    return {
        'loss': total_loss,
        'acceleration_loss': acc_loss,
        'stress_loss': stress_loss,
        'physics_loss': physics_loss,
    }


def _compute_physics_loss(
    self,
    predicted_acceleration,
    predicted_stress,
    inputs,
    sediment_mask
):
    """
    Compute physics-based loss constraints for granular materials.
    
    Constraints enforced:
    1. Stress tensor symmetry: σ_ij = σ_ji
    2. Pressure positivity: mean stress < 0 (compression positive convention)
    3. Deviatoric stress bounds: reasonable shear stress magnitudes
    
    Args:
        predicted_acceleration: [batch, n_particles, 3]
        predicted_stress: [batch, n_particles, 9]
        inputs: Input dictionary
        sediment_mask: [batch, n_particles] mask for sediment particles
    
    Returns:
        Scalar physics loss
    """
    # Reshape stress to 3x3 matrices [batch, n_particles, 3, 3]
    batch_size = tf.shape(predicted_stress)[0]
    n_particles = tf.shape(predicted_stress)[1]
    stress_3x3 = tf.reshape(predicted_stress, [batch_size, n_particles, 3, 3])
    
    # Expand mask for broadcasting
    mask_expanded = tf.expand_dims(tf.expand_dims(sediment_mask, -1), -1)
    
    # ========================================================================
    # Constraint 1: Stress tensor symmetry
    # ========================================================================
    # Stress tensor should be symmetric: σ_ij = σ_ji
    stress_transposed = tf.transpose(stress_3x3, [0, 1, 3, 2])
    symmetry_error = tf.abs(stress_3x3 - stress_transposed)
    symmetry_loss = tf.reduce_sum(symmetry_error * mask_expanded) / tf.reduce_sum(mask_expanded)
    
    # ========================================================================
    # Constraint 2: Pressure positivity (compression positive)
    # ========================================================================
    # Mean stress (pressure) = trace(σ) / 3
    trace = stress_3x3[:, :, 0, 0] + stress_3x3[:, :, 1, 1] + stress_3x3[:, :, 2, 2]
    mean_stress = trace / 3.0
    
    # For granular materials under gravity, we expect compression (negative mean stress)
    # Penalize positive mean stress (tension)
    pressure_violation = tf.nn.relu(mean_stress)  # max(0, mean_stress)
    pressure_loss = tf.reduce_sum(pressure_violation * sediment_mask) / tf.reduce_sum(sediment_mask)
    
    # ========================================================================
    # Constraint 3: Deviatoric stress bounds (optional)
    # ========================================================================
    # Compute deviatoric stress: s_ij = σ_ij - (1/3)δ_ij*trace(σ)
    identity = tf.eye(3, batch_shape=[batch_size, n_particles])
    mean_stress_3d = tf.expand_dims(tf.expand_dims(mean_stress, -1), -1)
    deviatoric = stress_3x3 - mean_stress_3d * identity
    
    # Von Mises stress (for yield criterion check)
    dev_squared = deviatoric ** 2
    von_mises = tf.sqrt(
        3.0 / 2.0 * tf.reduce_sum(dev_squared, axis=[2, 3])
    )
    
    # Soft constraint: penalize extremely large deviatoric stresses
    # (adjust threshold based on your material properties)
    max_deviatoric = 10000.0  # Pa or appropriate units
    deviatoric_violation = tf.nn.relu(von_mises - max_deviatoric)
    deviatoric_loss = tf.reduce_sum(deviatoric_violation * sediment_mask) / tf.reduce_sum(sediment_mask)
    
    # ========================================================================
    # Combine physics losses
    # ========================================================================
    physics_loss = (
        1.0 * symmetry_loss +      # Enforce symmetry strictly
        0.5 * pressure_loss +       # Encourage compression
        0.1 * deviatoric_loss       # Soft bound on shear
    )
    
    return physics_loss


def _rollout_with_stress(self, initial_positions, particle_types, n_particles_per_example,
                         n_steps, dt):
    """
    Perform rollout prediction with stress prediction (optional).
    
    This extends the standard rollout to also track stress evolution.
    Use for evaluation and visualization.
    
    Args:
        initial_positions: Initial particle positions
        particle_types: Particle type indices
        n_particles_per_example: Number of particles
        n_steps: Number of rollout steps
        dt: Timestep size
    
    Returns:
        Dictionary with:
            - 'positions': [n_steps+1, n_particles, 3]
            - 'velocities': [n_steps+1, n_particles, 3]
            - 'accelerations': [n_steps, n_particles, 3]
            - 'stress': [n_steps, n_particles, 9]
    """
    # Initialize trajectory storage
    positions = [initial_positions]
    velocities = [tf.zeros_like(initial_positions)]  # Assume zero initial velocity
    accelerations = []
    stresses = []
    
    # Current state
    current_position = initial_positions
    current_velocity = velocities[0]
    
    for step in range(n_steps):
        # Build position sequence (use last few timesteps for history)
        # For simplicity, just use current position
        position_sequence = tf.expand_dims(current_position, axis=1)  # Add seq dim
        
        # Predict next acceleration and stress
        predicted_acc, predicted_stress = self._predict_acceleration_and_stress(
            position_sequence=position_sequence,
            n_particles_per_example=n_particles_per_example,
            particle_types=particle_types,
        )
        
        # Store predictions
        accelerations.append(predicted_acc)
        stresses.append(predicted_stress)
        
        # Integrate with semi-implicit Euler
        # v_{t+1} = v_t + a_t * dt
        # x_{t+1} = x_t + v_{t+1} * dt
        next_velocity = current_velocity + predicted_acc * dt
        next_position = current_position + next_velocity * dt
        
        # Apply boundary conditions (particles with type=0 are fixed)
        is_boundary = tf.equal(particle_types, 0)
        is_boundary = tf.expand_dims(is_boundary, -1)  # [n_particles, 1]
        
        next_position = tf.where(is_boundary, initial_positions, next_position)
        next_velocity = tf.where(is_boundary, tf.zeros_like(next_velocity), next_velocity)
        
        # Update state
        current_position = next_position
        current_velocity = next_velocity
        
        positions.append(current_position)
        velocities.append(current_velocity)
    
    # Stack into arrays
    return {
        'positions': tf.stack(positions, axis=0),          # [n_steps+1, n_particles, 3]
        'velocities': tf.stack(velocities, axis=0),        # [n_steps+1, n_particles, 3]
        'accelerations': tf.stack(accelerations, axis=0),  # [n_steps, n_particles, 3]
        'stress': tf.stack(stresses, axis=0),              # [n_steps, n_particles, 9]
    }


# ============================================================================
# Example: Modified __init__ method for LearnedSimulator
# ============================================================================

def __init__(self, num_dimensions, connectivity_radius, graph_network_kwargs,
             boundaries, normalization_stats, num_particle_types,
             particle_type_embedding_size, use_stress_prediction=True):
    """
    Initialize LearnedSimulator with optional stress prediction.
    
    Args:
        ... (existing args) ...
        use_stress_prediction: If True, use dual decoder for stress prediction.
                              If False, use single decoder for acceleration only.
    """
    # ... existing initialization code ...
    
    self._use_stress_prediction = use_stress_prediction
    
    if use_stress_prediction:
        # Use dual decoder
        self._build_dual_decoder()
    else:
        # Use single decoder (original GNS)
        self._output_network = self._get_output_network(output_size=num_dimensions)


# ============================================================================
# Utility functions for stress tensor operations
# ============================================================================

def stress_to_components(stress_9d):
    """
    Convert 9D stress vector to named components.
    
    Args:
        stress_9d: [batch, n_particles, 9] tensor
    
    Returns:
        Dictionary with stress components
    """
    return {
        'sigma_xx': stress_9d[..., 0],
        'sigma_xy': stress_9d[..., 1],
        'sigma_xz': stress_9d[..., 2],
        'sigma_yx': stress_9d[..., 3],
        'sigma_yy': stress_9d[..., 4],
        'sigma_yz': stress_9d[..., 5],
        'sigma_zx': stress_9d[..., 6],
        'sigma_zy': stress_9d[..., 7],
        'sigma_zz': stress_9d[..., 8],
    }


def compute_von_mises_stress(stress_9d):
    """
    Compute von Mises stress from stress tensor.
    
    von Mises = sqrt(3/2 * s_ij * s_ij)
    where s_ij is deviatoric stress
    
    Args:
        stress_9d: [batch, n_particles, 9]
    
    Returns:
        von_mises: [batch, n_particles]
    """
    # Reshape to 3x3
    batch_size = tf.shape(stress_9d)[0]
    n_particles = tf.shape(stress_9d)[1]
    stress_3x3 = tf.reshape(stress_9d, [batch_size, n_particles, 3, 3])
    
    # Mean stress
    trace = stress_3x3[:, :, 0, 0] + stress_3x3[:, :, 1, 1] + stress_3x3[:, :, 2, 2]
    mean_stress = trace / 3.0
    
    # Deviatoric stress
    identity = tf.eye(3, batch_shape=[batch_size, n_particles])
    mean_stress_3d = tf.expand_dims(tf.expand_dims(mean_stress, -1), -1)
    deviatoric = stress_3x3 - mean_stress_3d * identity
    
    # von Mises
    dev_squared = deviatoric ** 2
    von_mises = tf.sqrt(3.0 / 2.0 * tf.reduce_sum(dev_squared, axis=[2, 3]))
    
    return von_mises


def compute_principal_stresses(stress_9d):
    """
    Compute principal stresses (eigenvalues of stress tensor).
    
    Args:
        stress_9d: [batch, n_particles, 9]
    
    Returns:
        principal_stresses: [batch, n_particles, 3] (sorted descending)
    """
    # Reshape to 3x3
    batch_size = tf.shape(stress_9d)[0]
    n_particles = tf.shape(stress_9d)[1]
    stress_3x3 = tf.reshape(stress_9d, [batch_size, n_particles, 3, 3])
    
    # Compute eigenvalues
    eigenvalues = tf.linalg.eigvalsh(stress_3x3)  # Returns sorted eigenvalues
    
    # Return in descending order (σ1 > σ2 > σ3)
    return tf.reverse(eigenvalues, axis=[-1])
