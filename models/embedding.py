import itertools
import math
from collections import Counter

import torch
import torch.nn as nn

from config import args
from dataloader.data_loader import DataCollectionLoader
from dataloader.data_processor import StateDataProcessor
from scripts.state_predict_script import prepare_dc_argument
from utils.visualization import BitEmbeddingVisualizer


class StateEmbeddingModel(nn.Module):
    """Neural network model for learning state embeddings."""
    def __init__(self, num_states: int, embedding_dim: int = 768):
        super().__init__()
        self.embedding = nn.Embedding(num_states, embedding_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices)


class SemanticBitEmbedding:
    def __init__(self):
        self.action_scores = {
            'wpc': 2,
            'wopc': 1,
            'abs': 0
        }  # TODO: extension for all actions
        self.bit_embeddings = None  # tuple key version
        self.labeled_bit_embeddings = None  # state label key version

    def create_labeled_embeddings(self, bit_embeddings, states, state_ids, processor):
        """Convert tuple-keyed embeddings to label-keyed embeddings"""
        labeled_embeddings = {}
        for state_tuple in states:
            state_id = state_ids[state_tuple]
            state_idx = list(processor.label_encoders['state'].classes_).index(state_id)
            state_label = processor.label_encoders['state'].inverse_transform([state_idx])[0]
            if state_tuple in bit_embeddings:
                labeled_embeddings[state_label] = bit_embeddings[state_tuple]

        return labeled_embeddings

    def calculate_state_score(self, state):
        return sum(self.action_scores[action] for action in state)

    def get_state_frequencies(self, y_train, processor, states, state_ids):
        """get state frequencies in train set"""
        train_counts = Counter(y_train)

        state_id_to_index = {state_id: idx for idx, state_id in
                             enumerate(processor.label_encoders['state'].classes_)}
        state_counts = {}
        for state_tuple in states:
            state_id = state_ids[state_tuple]
            if state_id in state_id_to_index:
                index = state_id_to_index[state_id]
                state_counts[state_tuple] = train_counts.get(int(index), 0)

        return state_counts

    def adjust_scores_by_frequency(self, states, state_ids, scores, y_train, processor):
        """Adjustment of scores by frequency in train set"""
        state_counts = self.get_state_frequencies(y_train, processor, states, state_ids)
        print(f'state_counts: {state_counts}')

        # group states has same score
        score_groups = {}
        for state, score in zip(states, scores):
            if score not in score_groups:
                score_groups[score] = []
            score_groups[score].append(state)
        print(f'score_groups: {score_groups}')

        adjusted_scores = {}
        for base_score, group_states in score_groups.items():
            if len(group_states) > 1:
                level_total = sum(state_counts.get(state, 0) for state in group_states)
                for state in set(group_states):
                    state_freq = state_counts.get(state, 0)
                    freq_ratio = state_freq / level_total if level_total > 0 else 0
                    adjusted_scores[state] = base_score + (1 - freq_ratio)
            else:  # 레벨에 하나의 상태만 있는 경우
                state = group_states[0]
                adjusted_scores[state] = base_score  # TODO: change to frequency?

        return [adjusted_scores[state] for state in states]

    def calculate_semantic_distances(self, states, adjusted_scores):
        """Calculate semantic distances between states"""
        distances = {}

        for state1, state2 in itertools.combinations(states, 2):
            score1 = adjusted_scores[states.index(state1)]
            score2 = adjusted_scores[states.index(state2)]
            distance = abs(score1 - score2)
            distances[(state1, state2)] = distance
            distances[(state2, state1)] = distance
        return distances

    def generate_bit_embeddings(self, states, semantic_distances, num_add=0):
        """Generate bit embeddings of states based on semantic distances"""
        num_states = len(states)
        num_bits = math.ceil(math.log2(num_states)) + num_add  # assign minimum bit representation + 1
        max_embeddings = 2 ** num_bits

        def hamming_distance(bits1, bits2):
            return bin(bits1 ^ bits2).count('1')

        bit_patterns = list(range(max_embeddings))
        sorted_distances = sorted(
            semantic_distances.items(),
            key=lambda x: x[1],
            reverse=True
        )

        embeddings = {}
        used_patterns = set()

        # start with the most far pair
        first_pair = sorted_distances[0][0]
        embeddings[first_pair[0]] = 0

        max_hamming = 0
        best_pattern = 1
        for pattern in bit_patterns[1:]:
            if pattern not in used_patterns:
                current_hamming = hamming_distance(0, pattern)
                if current_hamming > max_hamming:
                    max_hamming = current_hamming
                    best_pattern = pattern

        embeddings[first_pair[1]] = best_pattern
        used_patterns.add(0)
        used_patterns.add(best_pattern)

        remaining_states = set(states) - set(embeddings.keys())
        for state in remaining_states:
            best_pattern = None
            min_error = float('inf')

            for pattern in bit_patterns:
                if pattern not in used_patterns:
                    total_error = 0

                    for existing_state, existing_pattern in embeddings.items():
                        desired_distance = semantic_distances[(state, existing_state)]
                        actual_distance = hamming_distance(pattern, existing_pattern) / num_bits
                        total_error += (desired_distance - actual_distance) ** 2

                    if total_error < min_error:
                        min_error = total_error
                        best_pattern = pattern

            if best_pattern is not None:
                embeddings[state] = best_pattern
                used_patterns.add(best_pattern)

        return embeddings

    def process_states(self, total_states, state_ids, processor, num_add=0):
        """Total process"""
        y_train = processor.y_train

        # 1. calculate base scores
        base_scores = [self.calculate_state_score(state) for state in total_states]

        # 2. adjust scores by frequency
        adjusted_scores = self.adjust_scores_by_frequency(
            total_states, state_ids, base_scores, y_train, processor)
        print(f'adjusted score: {adjusted_scores}')
        # 3. calculate semantic distances
        semantic_distances = self.calculate_semantic_distances(total_states, adjusted_scores)

        # 4. generate bit embeddings based on semantic distances
        self.bit_embeddings = self.generate_bit_embeddings(total_states, semantic_distances, num_add)

        # 5. create labeled version of bit embeddings
        self.labeled_bit_embeddings = self.create_labeled_embeddings(
            self.bit_embeddings, total_states, state_ids, processor
        )

        return self.bit_embeddings, self.labeled_bit_embeddings


# ============================================================
if __name__ == '__main__':
    args.num_dc = 5
    dc_loader = DataCollectionLoader(args)
    processor = StateDataProcessor(args)
    df = dc_loader.load_preprocess()
    train_loader, val_loader, test_loader = processor.create_data_loaders(df)
    prepare_dc_argument(df, dc_loader, args)

    embedder = SemanticBitEmbedding()
    bit_embeddings, labeled_bit_embeddings = embedder.process_states(dc_loader.states, dc_loader.state_ids, processor, num_add=0)  # TODO: change bit length

    base_scores = {state: embedder.calculate_state_score(state) for state in dc_loader.states}
    adjusted_scores = {state: score for state, score in zip(dc_loader.states,
                                                            embedder.adjust_scores_by_frequency(dc_loader.states,
                                                                                                dc_loader.state_ids,
                                                                                                [base_scores[s] for s in
                                                                                                 dc_loader.states],
                                                                                                processor.y_train,
                                                                                                processor))}
    visualizer = BitEmbeddingVisualizer(bit_embeddings, dc_loader,
                                        scores=base_scores,
                                        adjusted_scores=adjusted_scores)
    visualizer.visualize_all()
