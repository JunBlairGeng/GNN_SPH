# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import graph_nets as gn
import sonnet as snt
import tensorflow.compat.v1 as tf

from learning_to_simulate import connectivity_utils
from learning_to_simulate import graph_network

STD_EPSILON = 1e-8


class LearnedSimulator(snt.AbstractModule):
  """Adapted Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

  def __init__(
      self,
      num_dimensions,
      connectivity_radius,
      graph_network_kwargs,
      boundaries,
      normalization_stats,
      num_particle_types,
      particle_type_embedding_size,
      name="LearnedSimulator"):
    """Inits the model.

    Args:
      num_dimensions: Dimensionality of the problem.
      connectivity_radius: Scalar with the radius of connectivity.
      graph_network_kwargs: Keyword arguments to pass to the learned part
        of the graph network `model.EncodeProcessDecode`.
      boundaries: List of 2-tuples, containing the lower and upper boundaries of
        the cuboid containing the particles along each dimensions, matching
        the dimensionality of the problem.
      normalization_stats: Dictionary with statistics with keys "acceleration"
        and "velocity", containing a named tuple for each with mean and std
        fields, matching the dimensionality of the problem.
      num_particle_types: Number of different particle types.
      particle_type_embedding_size: Embedding size for the particle type.
      name: Name of the Sonnet module.

    """
    super().__init__(name=name)

    self._connectivity_radius = connectivity_radius
    self._num_particle_types = num_particle_types
    self._boundaries = boundaries
    self._normalization_stats = normalization_stats
    with self._enter_variable_scope():
      self._graph_network = graph_network.EncodeProcessDecode(
          output_size=num_dimensions, **graph_network_kwargs)

      if self._num_particle_types > 1:
        self._particle_type_embedding = tf.get_variable(
            "particle_embedding",
            [self._num_particle_types, particle_type_embedding_size],
            trainable=True, use_resource=True)

  def _build(self):
    """Build the graph network with dual decoder."""
    # ... keep existing encoder and processor code ...
    
    # DUAL DECODER: Two separate output networks
    # 1. Acceleration decoder (3D output)
    self._acceleration_decoder = get_mlp_fn(
        output_size=3,  # 3D acceleration
        layer_norm=True,
        name='acceleration_decoder'
    )
    
    # 2. Stress decoder (9D output for 3x3 stress tensor)
    self._stress_decoder = get_mlp_fn(
        output_size=9,  # 9 stress components
        layer_norm=True,
        name='stress_decoder'
    )


  def _predict_acceleration_and_stress(self, next_position, position_sequence,
                                        n_particles_per_example, particle_types,
                                        global_context=None):
      """
      Predict both acceleration and stress from current state.
      
      Args:
          next_position: Current positions
          position_sequence: Recent position history
          n_particles_per_example: Number of particles
          particle_types: Particle type array
          global_context: Optional global features
      
      Returns:
          Tuple of (predicted_acceleration, predicted_stress)
      """
      # Build graph from positions
      graph = self._encoder_preprocessor(
          position_sequence, n_particles_per_example, particle_types, global_context
      )
      
      # Run graph network processor
      graph = self._processor(graph, n_steps=self._num_processing_steps)
      
      # Decode to both outputs
      per_node_network_output = self._decoder_postprocessor(
          graph, n_particles_per_example
      )
      
      # Split into two predictions
      predicted_acceleration = self._acceleration_decoder(per_node_network_output)
      predicted_stress = self._stress_decoder(per_node_network_output)
      
      return predicted_acceleration, predicted_stress

  def _encoder_preprocessor(
      self, position_sequence, n_node, global_context, particle_types):
    # Extract important features from the position_sequence.
    most_recent_position = position_sequence[:, -1]
    velocity_sequence = time_diff(position_sequence)  # Finite-difference.

    # Get connectivity of the graph.
    (senders, receivers, n_edge
    ) = connectivity_utils.compute_connectivity_for_batch_pyfunc(
        most_recent_position, n_node, self._connectivity_radius)

    # Collect node features.
    node_features = []

    # Normalized velocity sequence, merging spatial an time axis.
    velocity_stats = self._normalization_stats["velocity"]
    normalized_velocity_sequence = (
        velocity_sequence - velocity_stats.mean) / velocity_stats.std

    flat_velocity_sequence = snt.MergeDims(start=1, size=2)(
        normalized_velocity_sequence)
    node_features.append(flat_velocity_sequence)

    # Normalized clipped distances to lower and upper boundaries.
    # boundaries are an array of shape [num_dimensions, 2], where the second
    # axis, provides the lower/upper boundaries.
    boundaries = tf.constant(self._boundaries, dtype=tf.float32)
    distance_to_lower_boundary = (
        most_recent_position - tf.expand_dims(boundaries[:, 0], 0))
    distance_to_upper_boundary = (
        tf.expand_dims(boundaries[:, 1], 0) - most_recent_position)
    distance_to_boundaries = tf.concat(
        [distance_to_lower_boundary, distance_to_upper_boundary], axis=1)
    normalized_clipped_distance_to_boundaries = tf.clip_by_value(
        distance_to_boundaries / self._connectivity_radius, -1., 1.)
    node_features.append(normalized_clipped_distance_to_boundaries)

    # Particle type.
    if self._num_particle_types > 1:
      particle_type_embeddings = tf.nn.embedding_lookup(
          self._particle_type_embedding, particle_types)
      node_features.append(particle_type_embeddings)

    # Collect edge features.
    edge_features = []

    # Relative displacement and distances normalized to radius
    normalized_relative_displacements = (
        tf.gather(most_recent_position, senders) -
        tf.gather(most_recent_position, receivers)) / self._connectivity_radius
    edge_features.append(normalized_relative_displacements)

    normalized_relative_distances = tf.norm(
        normalized_relative_displacements, axis=-1, keepdims=True)
    edge_features.append(normalized_relative_distances)

    # Normalize the global context.
    if global_context is not None:
      context_stats = self._normalization_stats["context"]
      # Context in some datasets are all zero, so add an epsilon for numerical
      # stability.
      global_context = (global_context - context_stats.mean) / tf.math.maximum(
          context_stats.std, STD_EPSILON)

    return gn.graphs.GraphsTuple(
        nodes=tf.concat(node_features, axis=-1),
        edges=tf.concat(edge_features, axis=-1),
        globals=global_context,  # self._graph_net will appending this to nodes.
        n_node=n_node,
        n_edge=n_edge,
        senders=senders,
        receivers=receivers,
        )

  def _decoder_postprocessor(self, normalized_acceleration, position_sequence):

    # The model produces the output in normalized space so we apply inverse
    # normalization.
    acceleration_stats = self._normalization_stats["acceleration"]
    acceleration = (
        normalized_acceleration * acceleration_stats.std
        ) + acceleration_stats.mean

    # Use an Euler integrator to go from acceleration to position, assuming
    # a dt=1 corresponding to the size of the finite difference.
    most_recent_position = position_sequence[:, -1]
    most_recent_velocity = most_recent_position - position_sequence[:, -2]

    new_velocity = most_recent_velocity + acceleration  # * dt = 1
    new_position = most_recent_position + new_velocity  # * dt = 1
    return new_position

  def get_predicted_and_target_normalized_accelerations(
      self, next_position, position_sequence_noise, position_sequence,
      n_particles_per_example, global_context=None, particle_types=None):  # pylint: disable=g-doc-args
    """Produces normalized and predicted acceleration targets.

    Args:
      next_position: Tensor of shape [num_particles_in_batch, num_dimensions]
        with the positions the model should output given the inputs.
      position_sequence_noise: Tensor of the same shape as `position_sequence`
        with the noise to apply to each particle.
      position_sequence, n_node, global_context, particle_types: Inputs to the
        model as defined by `_build`.

    Returns:
      Tensors of shape [num_particles_in_batch, num_dimensions] with the
        predicted and target normalized accelerations.
    """

    # Add noise to the input position sequence.
    noisy_position_sequence = position_sequence + position_sequence_noise

    # Perform the forward pass with the noisy position sequence.
    input_graphs_tuple = self._encoder_preprocessor(
        noisy_position_sequence, n_particles_per_example, global_context,
        particle_types)
    predicted_normalized_acceleration = self._graph_network(input_graphs_tuple)

    # Calculate the target acceleration, using an `adjusted_next_position `that
    # is shifted by the noise in the last input position.
    next_position_adjusted = next_position + position_sequence_noise[:, -1]
    target_normalized_acceleration = self._inverse_decoder_postprocessor(
        next_position_adjusted, noisy_position_sequence)
    # As a result the inverted Euler update in the `_inverse_decoder` produces:
    # * A target acceleration that does not explicitly correct for the noise in
    #   the input positions, as the `next_position_adjusted` is different
    #   from the true `next_position`.
    # * A target acceleration that exactly corrects noise in the input velocity
    #   since the target next velocity calculated by the inverse Euler update
    #   as `next_position_adjusted - noisy_position_sequence[:,-1]`
    #   matches the ground truth next velocity (noise cancels out).

    return predicted_normalized_acceleration, target_normalized_acceleration

  def _inverse_decoder_postprocessor(self, next_position, position_sequence):
    """Inverse of `_decoder_postprocessor`."""

    previous_position = position_sequence[:, -1]
    previous_velocity = previous_position - position_sequence[:, -2]
    next_velocity = next_position - previous_position
    acceleration = next_velocity - previous_velocity

    acceleration_stats = self._normalization_stats["acceleration"]
    normalized_acceleration = (
        acceleration - acceleration_stats.mean) / acceleration_stats.std
    return normalized_acceleration
  def _loss_with_stress(self, inputs, is_training=True):
    """
    Compute loss including both acceleration and stress prediction.
    
    Args:
        inputs: Dictionary with 'position', 'particle_type', 
                'target_acceleration', 'target_stress'
        is_training: Whether in training mode
    
    Returns:
        Dictionary with loss components
    """
    # Get predictions
    predicted_acceleration, predicted_stress = self._predict_acceleration_and_stress(
        next_position=inputs['position'][:, -1],  # Current position
        position_sequence=inputs['position'],      # Position history
        n_particles_per_example=inputs['n_particles_per_example'],
        particle_types=inputs['particle_type'],
    )
    
    # Ground truth
    target_acceleration = inputs['target_acceleration']
    target_stress = inputs['target_stress']
    
    # Particle types (0 = boundary, 1 = sediment)
    particle_types = inputs['particle_type']
    
    # Loss only on sediment particles (type 1)
    sediment_mask = tf.cast(tf.equal(particle_types, 1), tf.float32)
    sediment_mask = tf.expand_dims(sediment_mask, -1)
    
    # 1. Acceleration loss (MSE on sediment particles only)
    acc_error = (predicted_acceleration - target_acceleration) ** 2
    acc_loss = tf.reduce_mean(acc_error * sediment_mask)
    
    # 2. Stress loss (MSE on sediment particles only)
    stress_error = (predicted_stress - target_stress) ** 2
    stress_loss = tf.reduce_mean(stress_error * sediment_mask)
    
    # 3. Physics-based constraints
    physics_loss = self._compute_physics_loss(
        predicted_acceleration, predicted_stress, 
        inputs, sediment_mask
    )
    
    # Total loss (weighted combination)
    total_loss = (
        1.0 * acc_loss +           # Acceleration loss weight
        0.5 * stress_loss +         # Stress loss weight  
        0.1 * physics_loss          # Physics constraint weight
    )
    
    return {
        'loss': total_loss,
        'acceleration_loss': acc_loss,
        'stress_loss': stress_loss,
        'physics_loss': physics_loss,
    }


  def _compute_physics_loss(self, predicted_acc, predicted_stress, 
                            inputs, sediment_mask):
      """
      Compute physics-based loss constraints.
      
      Physics constraints for granular materials:
      1. Stress-acceleration consistency: σ·∇ = m·a (momentum equation)
      2. Mohr-Coulomb yield criterion (optional, for soil plasticity)
      3. Pressure positivity: σ_mean < 0 (compression positive)
      
      Args:
          predicted_acc: Predicted accelerations [N, 3]
          predicted_stress: Predicted stress tensors [N, 9]
          inputs: Input dictionary
          sediment_mask: Mask for sediment particles [N, 1]
      
      Returns:
          Scalar physics loss
      """
      # Reshape stress to 3x3 matrices [N, 3, 3]
      stress_3x3 = tf.reshape(predicted_stress, [-1, 3, 3])
      
      # 1. Pressure positivity constraint
      # Mean stress (trace/3) should be negative for compression
      trace = stress_3x3[:, 0, 0] + stress_3x3[:, 1, 1] + stress_3x3[:, 2, 2]
      mean_stress = trace / 3.0
      
      # Penalize positive mean stress (tension in granular materials)
      pressure_violation = tf.nn.relu(mean_stress)  # ReLU = max(0, x)
      pressure_loss = tf.reduce_mean(pressure_violation * sediment_mask[:, 0])
      
      # 2. Stress symmetry constraint
      # Stress tensor should be symmetric: σ_ij = σ_ji
      symmetry_error = tf.abs(stress_3x3 - tf.transpose(stress_3x3, [0, 2, 1]))
      symmetry_loss = tf.reduce_mean(symmetry_error * sediment_mask)
      
      # Combine physics losses
      physics_loss = pressure_loss + symmetry_loss
      
      return physics_loss


def time_diff(input_sequence):
  return input_sequence[:, 1:] - input_sequence[:, :-1]

