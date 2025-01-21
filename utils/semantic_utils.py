from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from collections import defaultdict

import torch
from tabulate import tabulate


def create_semantic_groups() -> Dict[int, List[str]]:
    """Create semantic groups based on worker count."""
    return {
        2: ['S5'],  # 2 workers
        1: ['S7', 'S13', 'S6', 'S9', 'S4', 'S1'],  # 1 worker
        0: ['S3', 'S12', 'S8', 'S2', 'S0']  # 0 worker
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


def generate_binary_patterns(dim: int) -> List[np.ndarray]:
    """Generate all possible binary patterns for given dimension (2^dim patterns)."""
    patterns = []
    num_patterns = 2 ** dim  # Total number of possible patterns for dimension

    for i in range(num_patterns):
        pattern = np.array([int(x) for x in format(i, f'0{dim}b')])
        pattern = pattern * 2 - 1  # Convert to {-1, 1}
        patterns.append(pattern)
    return patterns


def find_most_similar_pattern(base_pattern: np.ndarray, available_patterns: List[np.ndarray]) -> np.ndarray:
    min_distance = float('inf')
    best_pattern = None

    for pattern in available_patterns:
        distance = np.sum(base_pattern != pattern)
        if distance < min_distance:
            min_distance = distance
            best_pattern = pattern

    return best_pattern


def create_hamming_bitwise_embedding(
        level_mapping: Dict[int, List[str]],
        num_bits: int,
        semantic_embeddings: Dict[str, np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """Create bit-wise embeddings."""
    all_states = []
    for level in sorted(level_mapping.keys()):
        all_states.extend(level_mapping[level])

    if semantic_embeddings is None:
        return create_binary_patterns_without_semantic(all_states, level_mapping, num_bits)

    return create_binary_patterns_with_semantic(all_states, num_bits, semantic_embeddings)


def create_binary_patterns_with_semantic(
        states: List[str],
        num_bits: int,
        semantic_embeddings: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Create binary patterns guided by semantic embeddings, allowing pattern reuse if necessary."""
    binary_embeddings = {}
    all_patterns = generate_binary_patterns(num_bits)
    num_patterns = len(all_patterns)

    # Calculate semantic similarities between all pairs
    semantic_similarities = {}
    for i, state1 in enumerate(states):
        for j, state2 in enumerate(states[i + 1:], i + 1):
            sim = cosine_similarity(
                semantic_embeddings[state1],
                semantic_embeddings[state2]
            )
            semantic_similarities[(state1, state2)] = sim
            semantic_similarities[(state2, state1)] = sim

    # If we don't have enough patterns, group states by semantic similarity
    if num_patterns < len(states):
        # Calculate average similarity for each state
        avg_similarities = {}
        for state in states:
            sims = [semantic_similarities[(state, other)] for other in states if other != state]
            avg_similarities[state] = np.mean(sims)

        # Sort states by average similarity
        sorted_states = sorted(states, key=lambda x: avg_similarities[x], reverse=True)

        # Create groups based on number of available patterns
        state_groups = []
        current_group = []

        for state in sorted_states:
            if not current_group:
                current_group = [state]
            else:
                # Check if this state is similar to the group
                group_similarities = [semantic_similarities[(state, s)] for s in current_group]
                avg_group_similarity = np.mean(group_similarities)

                if avg_group_similarity > 0.7 and len(current_group) < len(states) // num_patterns + 1:
                    current_group.append(state)
                else:
                    state_groups.append(current_group)
                    current_group = [state]

        if current_group:
            state_groups.append(current_group)

        # Assign patterns to groups
        for i, group in enumerate(state_groups):
            pattern = all_patterns[i % num_patterns]  # Reuse patterns if necessary
            for state in group:
                binary_embeddings[state] = pattern

    else:
        # We have enough patterns, assign unique patterns while preserving relationships
        available_pattern_indices = list(range(num_patterns))
        sorted_states = sorted(states,
                               key=lambda x: np.mean([semantic_similarities[(x, other)]
                                                      for other in states if other != x]),
                               reverse=True)

        for state in sorted_states:
            if not binary_embeddings:
                # First state gets a random pattern
                idx = np.random.choice(available_pattern_indices)
                binary_embeddings[state] = all_patterns[idx]
                available_pattern_indices.remove(idx)
            else:
                # Find best matching pattern based on semantic similarities
                best_pattern_idx = available_pattern_indices[0]
                best_score = float('-inf')

                for pattern_idx in available_pattern_indices:
                    pattern = all_patterns[pattern_idx]
                    score = 0

                    for other_state, other_pattern in binary_embeddings.items():
                        semantic_sim = semantic_similarities[(state, other_state)]
                        hamming_sim = calculate_pattern_similarity(pattern, other_pattern)
                        sim_diff = abs(semantic_sim - hamming_sim)
                        score -= sim_diff

                    if score > best_score:
                        best_score = score
                        best_pattern_idx = pattern_idx

                binary_embeddings[state] = all_patterns[best_pattern_idx]
                available_pattern_indices.remove(best_pattern_idx)

    return binary_embeddings


def calculate_pattern_similarity(pattern1: np.ndarray, pattern2: np.ndarray) -> float:
    """Calculate normalized similarity between two binary patterns."""
    return np.mean(pattern1 == pattern2)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


def create_binary_patterns_without_semantic(
        states: List[str],
        level_mapping: Dict[int, List[str]],
        num_bits: int
) -> Dict[str, np.ndarray]:
    """Create binary patterns without semantic guidance (fallback method)."""
    semantic_groups = create_semantic_groups()
    num_states = len(states)
    available_patterns = 2 ** num_bits

    if available_patterns < num_states:
        return create_shared_patterns(states, semantic_groups, level_mapping, num_bits)
    else:
        return create_unique_patterns(states, semantic_groups, level_mapping, num_bits)


def create_shared_patterns(
        states: List[str],
        semantic_groups: Dict[int, List[str]],
        level_mapping: Dict[int, List[str]],
        num_bits: int
) -> Dict[str, np.ndarray]:
    """Create patterns where semantically similar states share patterns."""
    binary_embeddings = {}
    all_patterns = generate_binary_patterns(num_bits)  # 2^num_bits patterns

    # Group states by semantic similarity and level
    state_groups = defaultdict(list)
    for state in states:
        level, worker_count = get_state_info(state, level_mapping, semantic_groups)
        state_groups[(level, worker_count)].append(state)

    # Assign patterns to groups
    pattern_idx = 0
    available_patterns = len(all_patterns)

    for group_states in state_groups.values():
        # Use modulo to wrap around if we run out of patterns
        current_pattern = all_patterns[pattern_idx % available_patterns]
        pattern_idx += 1

        # Assign same pattern to all states in group
        for state in group_states:
            binary_embeddings[state] = current_pattern

    return binary_embeddings


def create_unique_patterns(
        states: List[str],
        semantic_groups: Dict[int, List[str]],
        level_mapping: Dict[int, List[str]],
        num_bits: int
) -> Dict[str, np.ndarray]:
    """Create unique patterns for each state."""
    binary_embeddings = {}
    all_patterns = generate_binary_patterns(num_bits)  # 2^num_bits patterns

    # First, group states by semantic similarity
    state_groups = defaultdict(list)
    for state in states:
        level, worker_count = get_state_info(state, level_mapping, semantic_groups)
        state_groups[(level, worker_count)].append(state)

    used_pattern_indices = set()

    # Assign patterns to states within each semantic group
    for (level, worker_count), group_states in state_groups.items():
        # Try to assign similar patterns to states in same group
        base_pattern_idx = None
        for state in group_states:
            if base_pattern_idx is None:
                # Find first unused pattern for this group
                for idx in range(len(all_patterns)):
                    if idx not in used_pattern_indices:
                        base_pattern_idx = idx
                        used_pattern_indices.add(idx)
                        binary_embeddings[state] = all_patterns[idx]
                        break
            else:
                # Find the closest unused pattern to base pattern
                min_distance = float('inf')
                best_idx = None
                base_pattern = all_patterns[base_pattern_idx]

                for idx in range(len(all_patterns)):
                    if idx not in used_pattern_indices:
                        distance = np.sum(base_pattern != all_patterns[idx])
                        if distance < min_distance:
                            min_distance = distance
                            best_idx = idx

                used_pattern_indices.add(best_idx)
                binary_embeddings[state] = all_patterns[best_idx]

    return binary_embeddings


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


def analyze_pattern_uniqueness(
        bitwise_embeddings: Dict[str, np.ndarray],
        level_mapping: Dict[int, List[str]],
        semantic_groups: Dict[int, List[str]]
) -> bool:
    """Analyze if patterns are unique and print detailed information."""
    # Get dimension and pattern information
    num_states = len(bitwise_embeddings)
    pattern_dim = len(next(iter(bitwise_embeddings.values())))
    possible_patterns = 2 ** pattern_dim

    print("\n=== Pattern Analysis ===")
    print(f"Number of states: {num_states}")
    print(f"Embedding dimension: {pattern_dim}")
    print(f"Possible patterns (2^dim): {possible_patterns}")

    # Check for unique patterns
    pattern_to_states = defaultdict(list)
    for state, pattern in bitwise_embeddings.items():
        pattern_tuple = tuple(pattern.tolist())
        pattern_to_states[pattern_tuple].append(state)

    num_unique_patterns = len(pattern_to_states)
    is_unique = num_unique_patterns == num_states

    print(f"Number of unique patterns used: {num_unique_patterns}")
    print(f"All patterns are unique: {is_unique}")

    if not is_unique:
        print("\nStates sharing patterns:")
        for pattern, states in pattern_to_states.items():
            if len(states) > 1:
                print(f"\nPattern: {pattern[:10]}...")
                print("States:", end=" ")
                for state in states:
                    level, workers = get_state_info(state, level_mapping, semantic_groups)
                    print(f"{state}(Level:{level}, Workers:{workers})", end=" ")
                print()

    return is_unique


def print_embeddings(semantic_embeddings: Dict[str, np.ndarray], bitwise_embeddings: Dict[str, np.ndarray],
                     level_mapping: Dict[int, List[str]], args=None):
    """Print embeddings in a table format with limited dimensions."""
    analyze_pattern_uniqueness(bitwise_embeddings, level_mapping, create_semantic_groups())

    table_data = []
    num_dims = args.num_dims if args and hasattr(args, 'num_dims') else 4

    # Add header row
    headers = ['Level', 'State']
    headers.extend([f'Sem_{i}' for i in range(num_dims)])
    headers.extend(['...'])  # Separator
    headers.extend([f'Bit_{i}' for i in range(num_dims)])
    headers.extend(['...'])

    # Add data rows
    for level, states in sorted(level_mapping.items()):
        for state in states:
            row = [
                level,
                state,
                *semantic_embeddings[state][:num_dims].round(4),
                '...',  # Separator
                *bitwise_embeddings[state][:num_dims],
                '...'
            ]
            table_data.append(row)

    print("\nEmbedding Values:")
    print(tabulate(table_data, headers=headers, tablefmt='grid'))


if __name__ == '__main__':
    dim = 4
    patterns = generate_binary_patterns(dim)
    print(patterns)

    pattern_to_remove = np.array([-1, 1, 1, -1], dtype=np.float32)
    patterns = [p for p in patterns if not np.array_equal(p, pattern_to_remove)]
    best_pattern = find_most_similar_pattern(np.array([-1, 1, 1, -1]), patterns)
    print(best_pattern)
