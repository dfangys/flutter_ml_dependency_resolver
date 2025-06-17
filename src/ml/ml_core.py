"""
Machine Learning core for Flutter dependency resolution.
Implements reinforcement learning and graph neural networks for intelligent dependency selection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import pickle
from pathlib import Path
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib


# Experience tuple for reinforcement learning
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


@dataclass
class MLConfig:
    """Configuration for ML models."""
    # RL Agent parameters
    state_dim: int = 128
    action_dim: int = 100  # Max number of version choices per dependency
    hidden_dim: int = 256
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 32
    target_update_freq: int = 100
    
    # GNN parameters
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 3
    gnn_dropout: float = 0.1
    
    # Training parameters
    max_episodes: int = 1000
    max_steps_per_episode: int = 50
    save_freq: int = 100


class DependencyStateEncoder:
    """Encodes dependency graph state into feature vectors for ML models."""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.package_vocab = {}  # Maps package names to indices
        self.version_vocab = {}  # Maps version strings to indices
        self.is_fitted = False
    
    def fit(self, dependency_graphs: List[Any]):
        """Fit the encoder on a collection of dependency graphs."""
        # Build vocabulary from all graphs
        all_packages = set()
        all_versions = set()
        
        for graph in dependency_graphs:
            for dep_name, dep_constraint in graph.get_all_dependencies().items():
                all_packages.add(dep_name)
                all_versions.add(dep_constraint.constraint)
        
        # Create vocabularies
        self.package_vocab = {pkg: idx for idx, pkg in enumerate(sorted(all_packages))}
        self.version_vocab = {ver: idx for idx, ver in enumerate(sorted(all_versions))}
        
        # Fit scaler on sample features
        sample_features = []
        for graph in dependency_graphs[:100]:  # Use subset for efficiency
            features = self._extract_graph_features(graph)
            sample_features.append(features)
        
        if sample_features:
            self.scaler.fit(sample_features)
            self.is_fitted = True
    
    def encode_state(self, dependency_graph: Any, current_selection: Dict[str, str] = None) -> np.ndarray:
        """Encode dependency graph state into feature vector."""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before encoding states")
        
        features = self._extract_graph_features(dependency_graph, current_selection)
        return self.scaler.transform([features])[0]
    
    def _extract_graph_features(self, graph: Any, current_selection: Dict[str, str] = None) -> List[float]:
        """Extract numerical features from dependency graph."""
        features = []
        
        # Basic graph statistics
        all_deps = graph.get_all_dependencies()
        features.extend([
            len(all_deps),  # Total number of dependencies
            len(graph.dependencies),  # Regular dependencies
            len(graph.dev_dependencies),  # Dev dependencies
            len(graph.dependency_overrides),  # Overrides
            len(graph.project_metadata.platforms),  # Number of platforms
        ])
        
        # Graph topology features
        if graph.graph.number_of_nodes() > 0:
            features.extend([
                graph.graph.number_of_nodes(),
                graph.graph.number_of_edges(),
                nx.density(graph.graph) if graph.graph.number_of_nodes() > 1 else 0,
                len(list(nx.weakly_connected_components(graph.graph))),
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Constraint type distribution
        constraint_types = {'caret': 0, 'range': 0, 'exact': 0, 'any': 0}
        for dep in all_deps.values():
            constraint_types[dep.constraint_type] = constraint_types.get(dep.constraint_type, 0) + 1
        
        total_deps = len(all_deps) if all_deps else 1
        features.extend([count / total_deps for count in constraint_types.values()])
        
        # Platform-specific features
        platform_counts = {platform: 0 for platform in ['android', 'ios', 'web', 'windows', 'macos', 'linux']}
        for dep in all_deps.values():
            if dep.platform_specific and dep.platform_specific in platform_counts:
                platform_counts[dep.platform_specific] += 1
        
        features.extend([count / total_deps for count in platform_counts.values()])
        
        # Current selection features (if provided)
        if current_selection:
            selection_features = self._extract_selection_features(all_deps, current_selection)
            features.extend(selection_features)
        else:
            # Pad with zeros if no current selection
            features.extend([0] * 10)
        
        # Pad or truncate to fixed size
        target_size = self.config.state_dim
        if len(features) < target_size:
            features.extend([0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return features
    
    def _extract_selection_features(self, all_deps: Dict[str, Any], current_selection: Dict[str, str]) -> List[float]:
        """Extract features related to current dependency selection."""
        features = []
        
        # Selection completeness
        selected_count = len(current_selection)
        total_count = len(all_deps)
        features.append(selected_count / total_count if total_count > 0 else 0)
        
        # Constraint satisfaction rate
        satisfied_count = 0
        for dep_name, selected_version in current_selection.items():
            if dep_name in all_deps:
                if all_deps[dep_name].satisfies_version(selected_version):
                    satisfied_count += 1
        
        features.append(satisfied_count / selected_count if selected_count > 0 else 0)
        
        # Add more selection-specific features
        features.extend([0] * 8)  # Placeholder for additional features
        
        return features


class DQNAgent(nn.Module):
    """Deep Q-Network agent for dependency resolution."""
    
    def __init__(self, config: MLConfig):
        super(DQNAgent, self).__init__()
        self.config = config
        
        # Neural network layers
        self.fc1 = nn.Linear(config.state_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.fc3 = nn.Linear(config.hidden_dim, config.action_dim)
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() < epsilon:
            return random.randint(0, self.config.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.forward(state_tensor)
            return q_values.argmax().item()


class ReplayMemory:
    """Experience replay memory for DQN training."""
    
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        """Add experience to memory."""
        self.memory.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences."""
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        return len(self.memory)


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for dependency relationship modeling."""
    
    def __init__(self, config: MLConfig):
        super(GraphNeuralNetwork, self).__init__()
        self.config = config
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        input_dim = config.gnn_hidden_dim
        
        for i in range(config.gnn_num_layers):
            self.conv_layers.append(
                nn.Linear(input_dim, config.gnn_hidden_dim)
            )
        
        self.dropout = nn.Dropout(config.gnn_dropout)
        self.output_layer = nn.Linear(config.gnn_hidden_dim, config.gnn_hidden_dim)
    
    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """Forward pass through GNN."""
        x = node_features
        
        for conv_layer in self.conv_layers:
            # Graph convolution: A * X * W
            x = torch.matmul(adjacency_matrix, x)
            x = conv_layer(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.output_layer(x)
        return x


class DependencyResolutionAgent:
    """Main ML agent for dependency resolution combining DQN and GNN."""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.state_encoder = DependencyStateEncoder(config)
        self.dqn = DQNAgent(config)
        self.target_dqn = DQNAgent(config)
        self.gnn = GraphNeuralNetwork(config)
        self.memory = ReplayMemory(config.memory_size)
        
        # Training components
        self.optimizer = optim.Adam(
            list(self.dqn.parameters()) + list(self.gnn.parameters()),
            lr=config.learning_rate
        )
        self.criterion = nn.MSELoss()
        
        # Training state
        self.epsilon = config.epsilon_start
        self.episode_count = 0
        self.step_count = 0
        
        # Copy weights to target network
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_dqn.load_state_dict(self.dqn.state_dict())
    
    def train_step(self) -> float:
        """Perform one training step."""
        if len(self.memory) < self.config.batch_size:
            return 0.0
        
        # Sample batch from memory
        batch = self.memory.sample(self.config.batch_size)
        
        # Prepare batch tensors
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch])
        
        # Current Q values
        current_q_values = self.dqn(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_dqn(next_states).max(1)[0]
            target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.config.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
        
        return loss.item()
    
    def select_dependency_version(self, dependency_graph: Any, dependency_name: str, 
                                available_versions: List[str]) -> str:
        """Select optimal version for a dependency using ML."""
        # Encode current state
        state = self.state_encoder.encode_state(dependency_graph)
        
        # Get action from DQN
        action_idx = self.dqn.select_action(state, self.epsilon)
        
        # Map action to version selection
        if available_versions:
            version_idx = action_idx % len(available_versions)
            return available_versions[version_idx]
        else:
            return "latest"  # Fallback
    
    def evaluate_resolution(self, dependency_graph: Any, resolution: Dict[str, str]) -> float:
        """Evaluate the quality of a dependency resolution."""
        score = 0.0
        
        # Check constraint satisfaction
        all_deps = dependency_graph.get_all_dependencies()
        satisfied_count = 0
        
        for dep_name, selected_version in resolution.items():
            if dep_name in all_deps:
                if all_deps[dep_name].satisfies_version(selected_version):
                    satisfied_count += 1
        
        # Constraint satisfaction score (0-1)
        constraint_score = satisfied_count / len(resolution) if resolution else 0
        score += constraint_score * 0.4
        
        # Completeness score
        completeness_score = len(resolution) / len(all_deps) if all_deps else 0
        score += completeness_score * 0.3
        
        # Stability score (prefer stable versions)
        stability_score = self._calculate_stability_score(resolution)
        score += stability_score * 0.3
        
        return score
    
    def _calculate_stability_score(self, resolution: Dict[str, str]) -> float:
        """Calculate stability score based on version choices."""
        # Simplified stability scoring
        # In practice, this would consider factors like:
        # - Version age and adoption
        # - Known issues and vulnerabilities
        # - Community feedback
        
        stable_count = 0
        for version in resolution.values():
            # Simple heuristic: prefer non-prerelease versions
            if not any(marker in version.lower() for marker in ['alpha', 'beta', 'rc', 'dev']):
                stable_count += 1
        
        return stable_count / len(resolution) if resolution else 0
    
    def save_model(self, path: Path):
        """Save trained model to disk."""
        model_data = {
            'dqn_state_dict': self.dqn.state_dict(),
            'target_dqn_state_dict': self.target_dqn.state_dict(),
            'gnn_state_dict': self.gnn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'state_encoder': self.state_encoder
        }
        
        torch.save(model_data, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path):
        """Load trained model from disk."""
        model_data = torch.load(path, map_location='cpu')
        
        self.dqn.load_state_dict(model_data['dqn_state_dict'])
        self.target_dqn.load_state_dict(model_data['target_dqn_state_dict'])
        self.gnn.load_state_dict(model_data['gnn_state_dict'])
        self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
        
        self.epsilon = model_data['epsilon']
        self.episode_count = model_data['episode_count']
        self.step_count = model_data['step_count']
        self.state_encoder = model_data['state_encoder']
        
        self.logger.info(f"Model loaded from {path}")


class EnsembleResolver:
    """Ensemble of different ML models for robust dependency resolution."""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize ensemble members
        self.rl_agent = DependencyResolutionAgent(config)
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ensemble_weights = [0.6, 0.4]  # Weights for RL agent and RF
        
        self.is_trained = False
    
    def train_ensemble(self, training_data: List[Tuple[Any, Dict[str, str], float]]):
        """Train the ensemble on historical resolution data."""
        # Training data format: (dependency_graph, resolution, success_score)
        
        # Prepare data for Random Forest
        rf_features = []
        rf_labels = []
        
        for graph, resolution, score in training_data:
            features = self.rl_agent.state_encoder.encode_state(graph)
            rf_features.append(features)
            rf_labels.append(1 if score > 0.8 else 0)  # Binary classification
        
        if rf_features:
            self.rf_classifier.fit(rf_features, rf_labels)
            self.is_trained = True
            self.logger.info("Ensemble training completed")
    
    def resolve_dependencies(self, dependency_graph: Any, 
                           available_versions: Dict[str, List[str]]) -> Dict[str, str]:
        """Resolve dependencies using ensemble approach."""
        resolution = {}
        
        for dep_name, versions in available_versions.items():
            if not versions:
                continue
            
            # Get predictions from ensemble members
            rl_version = self.rl_agent.select_dependency_version(
                dependency_graph, dep_name, versions
            )
            
            # RF prediction (simplified - would need more sophisticated integration)
            rf_confidence = 0.5
            if self.is_trained:
                state = self.rl_agent.state_encoder.encode_state(dependency_graph)
                rf_prob = self.rf_classifier.predict_proba([state])[0]
                rf_confidence = max(rf_prob)
            
            # Weighted ensemble decision
            if rf_confidence > 0.7:
                resolution[dep_name] = rl_version
            else:
                # Fallback to conservative choice
                resolution[dep_name] = versions[0] if versions else "latest"
        
        return resolution


def setup_ml_logging():
    """Setup logging for ML components."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


if __name__ == "__main__":
    setup_ml_logging()
    
    # Example usage
    config = MLConfig()
    agent = DependencyResolutionAgent(config)
    
    print("ML components initialized successfully")
    print(f"State dimension: {config.state_dim}")
    print(f"Action dimension: {config.action_dim}")
    print(f"Hidden dimension: {config.hidden_dim}")

