from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.preprocessing import MinMaxScaler, normalize


def create_semantic_groups() -> Dict[int, List[str]]:
    """Create semantic groups based on worker count."""
    return {
        2: ['S5'],  # 2명 작업
        1: ['S7', 'S13', 'S6', 'S9', 'S4', 'S1'],  # 1명 작업
        0: ['S3', 'S12', 'S8', 'S2', 'S0']  # 0명 작업
    }


def get_state_info(
        state: str,
        level_mapping: Dict[int, List[str]],
        semantic_groups: Dict[int, List[str]]
) -> Tuple[Optional[int], Optional[int]]:
    """Get level and worker count information for a state."""
    level = next((level for level, states in level_mapping.items()
                  if state in states), None)
    worker_count = next((count for count, states in semantic_groups.items()
                         if state in states), None)
    return level, worker_count


def create_training_pairs(level_mapping: Dict[int, List[str]]) -> Tuple[
    Dict[str, int], List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int, float]]]:
    """Create trainer pairs for embedding model."""
    # Create state to index mapping
    all_states = []
    for level in sorted(level_mapping.keys()):
        all_states.extend(level_mapping[level])
    state_to_idx = {state: idx for idx, state in enumerate(all_states)}

    semantic_groups = create_semantic_groups()
    max_level_diff = max(level_mapping.keys()) - min(level_mapping.keys())
    max_worker_diff = 2

    positive_pairs = []  # Same level pairs
    semantic_pairs = []  # Same worker count pairs
    negative_pairs = []  # Different level and worker count pairs

    states = list(state_to_idx.keys())
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            state1, state2 = states[i], states[j]
            idx1, idx2 = state_to_idx[state1], state_to_idx[state2]

            level1, worker_count1 = get_state_info(state1, level_mapping, semantic_groups)
            level2, worker_count2 = get_state_info(state2, level_mapping, semantic_groups)

            level_diff = abs(level1 - level2) / max_level_diff
            worker_diff = abs(worker_count1 - worker_count2) / max_worker_diff

            if level1 == level2:
                positive_pairs.append((idx1, idx2))

            if worker_count1 == worker_count2:
                semantic_pairs.append((idx1, idx2))

            similarity_target = 1.0 - (0.6 * level_diff + 0.4 * worker_diff)
            negative_pairs.append((idx1, idx2, similarity_target))

    return state_to_idx, positive_pairs, semantic_pairs, negative_pairs


def create_hamming_bitwise_embedding(
        semantic_embeddings: Dict[str, np.ndarray],
        level_mapping: Dict[int, List[str]],
        num_bits: int = 4
) -> Dict[str, np.ndarray]:
    """Create bit-wise embeddings from semantic embeddings."""
    states = sorted(semantic_embeddings.keys())
    n_states = len(states)

    # Calculate similarity matrix
    similarity_matrix = np.zeros((n_states, n_states))
    for i, state1 in enumerate(states):
        for j, state2 in enumerate(states):
            vec1, vec2 = semantic_embeddings[state1], semantic_embeddings[state2]
            similarity_matrix[i, j] = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # Normalize to [0,1] range
    similarity_matrix = (similarity_matrix + 1) / 2
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_features = scaler.fit_transform(similarity_matrix)

    # Create binary embeddings
    thresholds = np.linspace(0, 1, num_bits + 1)[1:-1]
    return {
        state: np.array([1 if feat > thresh else -1
                         for feat in normalized_features[i]
                         for thresh in thresholds])
        for i, state in enumerate(states)
    }
