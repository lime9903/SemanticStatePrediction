from typing import Dict, Tuple, List
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.embedding import StateEmbeddingModel
from utils.semantic_utils import create_training_pairs
from sklearn.preprocessing import normalize


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_embedding_model(
        level_mapping: Dict[int, List[str]],
        embedding_dim: int = 768,
        num_epochs: int = 300,
        learning_rate: float = 0.005
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Train the embedding model and return state embeddings."""
    set_seeds(100)
    state_to_idx, positive_pairs, semantic_pairs, negative_pairs = create_training_pairs(level_mapping)

    # Initialize model and optimizer
    model = StateEmbeddingModel(len(state_to_idx), embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    cos = nn.CosineSimilarity(dim=1)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Calculate losses for different pair types
        level_loss = _calculate_level_loss(model, positive_pairs, cos) if positive_pairs else torch.tensor(0.0)
        semantic_loss = _calculate_semantic_loss(model, semantic_pairs, cos) if semantic_pairs else torch.tensor(0.0)
        neg_loss = _calculate_negative_loss(model, negative_pairs, cos) if negative_pairs else torch.tensor(0.0)

        # Combined loss
        loss = 2.0 * level_loss + 1.5 * semantic_loss + neg_loss

        if epoch % 50 == 0:
            _print_training_status(epoch, loss, level_loss, semantic_loss)

        loss.backward()
        optimizer.step()

    # Generate final embeddings
    return _generate_embeddings(model, state_to_idx)


def _calculate_level_loss(
        model: nn.Module,
        positive_pairs: List[Tuple[int, int]],
        cos: nn.CosineSimilarity
) -> torch.Tensor:
    """Calculate loss for same-level pairs."""
    pos_indices = torch.tensor([[p[0], p[1]] for p in positive_pairs])
    pos_embeddings = model(pos_indices)
    pos_similarity = cos(pos_embeddings[:, 0], pos_embeddings[:, 1])
    return torch.mean((1.0 - pos_similarity).pow(2))


def _calculate_semantic_loss(
        model: nn.Module,
        semantic_pairs: List[Tuple[int, int]],
        cos: nn.CosineSimilarity
) -> torch.Tensor:
    """Calculate loss for semantic pairs."""
    sem_indices = torch.tensor([[p[0], p[1]] for p in semantic_pairs])
    sem_embeddings = model(sem_indices)
    sem_similarity = cos(sem_embeddings[:, 0], sem_embeddings[:, 1])
    return torch.mean((0.8 - sem_similarity).pow(2))


def _calculate_negative_loss(
        model: nn.Module,
        negative_pairs: List[Tuple[int, int, float]],
        cos: nn.CosineSimilarity
) -> torch.Tensor:
    """Calculate loss for negative pairs."""
    neg_indices = torch.tensor([[p[0], p[1]] for p in negative_pairs])
    neg_targets = torch.tensor([p[2] for p in negative_pairs])
    neg_embeddings = model(neg_indices)
    neg_similarity = cos(neg_embeddings[:, 0], neg_embeddings[:, 1])
    return torch.mean((neg_targets - neg_similarity).pow(2))


def _print_training_status(
        epoch: int,
        loss: torch.Tensor,
        level_loss: torch.Tensor,
        semantic_loss: torch.Tensor
) -> None:
    """Print trainer status."""
    print(f"Epoch {epoch}")
    print(f"Total Loss: {loss.item():.4f}")
    if level_loss > 0:
        print(f"Level Loss: {level_loss.item():.4f}")
    if semantic_loss > 0:
        print(f"Semantic Loss: {semantic_loss.item():.4f}")
    print("---")


def _generate_embeddings(
        model: nn.Module,
        state_to_idx: Dict[str, int]
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Generate final embeddings from trained model."""
    with torch.no_grad():
        max_idx = max(state_to_idx.values())
        indices = torch.tensor(range(max_idx + 1))
        embeddings = model(indices).numpy()
        normalized_embeddings = normalize(embeddings)

    idx_to_state = {v: k for k, v in state_to_idx.items()}
    state_embeddings = {
        state: normalized_embeddings[idx]
        for state, idx in state_to_idx.items()
    }

    return state_embeddings, state_to_idx
