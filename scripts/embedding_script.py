from typing import Dict, List

from trainer.embedding_trainer import train_embedding_model
from utils.semantic_utils import create_hamming_bitwise_embedding, print_embeddings
from utils.visualization import analyze_embeddings
from config import args


def main() -> None:
    """Main function to run the embedding analysis pipeline."""
    # Configuration
    args.embedding_dim = 4
    args.num_bits = 4
    args.embed_nepochs = 800
    args.num_dims = 8

    # Define level mapping
    level_mapping: Dict[int, List[str]] = {
        0: ['S5'],
        1: ['S7', 'S13'],
        2: ['S6', 'S9'],
        3: ['S4', 'S1'],
        4: ['S3', 'S12'],
        5: ['S8', 'S2'],
        6: ['S0']
    }

    # Train semantic embeddings
    semantic_embeddings, state_to_idx = train_embedding_model(
        level_mapping,
        embedding_dim=args.embedding_dim,
        num_epochs=args.embed_nepochs
    )

    # Create bit-wise embeddings
    bitwise_embeddings = create_hamming_bitwise_embedding(
        level_mapping,
        num_bits=args.num_bits,
        semantic_embeddings=semantic_embeddings
    )

    # Analyze and visualize embeddings
    correlation, semantic_sim, hamming_dist = analyze_embeddings(
        semantic_embeddings,
        bitwise_embeddings,
        level_mapping
    )

    print(f"Final correlation between semantic and bitwise embeddings: {correlation:.3f}")
    # Print embeddings in table format (show first 16 dimensions)
    print_embeddings(semantic_embeddings, bitwise_embeddings, level_mapping, args)


if __name__ == "__main__":
    main()
