import torch
import torch.nn as nn


class StateEmbeddingModel(nn.Module):
    """Neural network model for learning state embeddings."""
    def __init__(self, num_states: int, embedding_dim: int = 768):
        super().__init__()
        self.embedding = nn.Embedding(num_states, embedding_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices)
