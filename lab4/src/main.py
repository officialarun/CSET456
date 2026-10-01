import time

from data_loader import (
    load_combined_dataset,
    load_token_frequency,
    load_bpe_tokenizer
)

from token_selection import (
    select_tokens
)

from frequency_embedding import (
    generate_frequency_embeddings
)

from position_embedding import (
    generate_position_embeddings
)

from similarity_analysis import (
    calculate_pairwise_similarity
)

from word2vec_embedding import (
    train_word2vec
)

from code2vec_embedding import (
    generate_code2vec_embeddings
)

from comparison import (
    compare_similarity_results
)


def print_header():

    print("\n")
    print("=" * 78)
    print("                    SPECIAL TOPICS IN DEVOPS")
    print("                             LAB - 4")
    print("                 SOFTWARE REPOSITORY EMBEDDINGS")
    print("=" * 78)

    print("\nPipeline:")
    print(
        "Data → Token Selection → Custom Embeddings → "
        "Word2Vec → Code2Vec → Similarity → Comparison"
    )


def main():

    start_time = time.time()

    print_header()

    try:

        # ====================================================
        # STEP 1
        # ====================================================

        print("\n")
        print("#" * 78)
        print("# STEP 1 — LOAD DATA")
        print("#" * 78)

        combined_dataset = load_combined_dataset()
        token_frequency = load_token_frequency()
        tokenizer = load_bpe_tokenizer()

        # ====================================================
        # STEP 2
        # ====================================================

        print("\n")
        print("#" * 78)
        print("# STEP 2 — SELECT 20 TOKENS")
        print("#" * 78)

        selected_tokens = select_tokens(token_frequency, number_of_tokens=20)

        print("\n[✓] Manual token selection completed.")

        # ====================================================
        # STEP 3
        # ====================================================

        print("\n")
        print("#" * 78)
        print("# STEP 3 — FREQUENCY EMBEDDING")
        print("#" * 78)

        frequency_embeddings = generate_frequency_embeddings(
            selected_tokens,
            token_frequency,
            combined_dataset
        )

        frequency_similarity = calculate_pairwise_similarity(
            selected_tokens,
            frequency_embeddings,
            "frequency_embedding"
        )

        print("\n[✓] Frequency embedding experiment completed.")

        # ====================================================
        # STEP 4
        # ====================================================

        print("\n")
        print("#" * 78)
        print("# STEP 4 — POSITION / WORD-FLOW EMBEDDING")
        print("#" * 78)

        position_embeddings = generate_position_embeddings(
            selected_tokens,
            combined_dataset,
            tokenizer
        )

        position_similarity = calculate_pairwise_similarity(
            selected_tokens,
            position_embeddings,
            "position_embedding"
        )

        print("\n[✓] Position embedding experiment completed.")

        # ====================================================
        # STEP 5
        # ====================================================

        print("\n")
        print("#" * 78)
        print("# STEP 5 — WORD2VEC")
        print("#" * 78)

        (
            word2vec_tokens,
            word2vec_embeddings,
            word2vec_model
        ) = train_word2vec(
            selected_tokens,
            combined_dataset,
            tokenizer
        )

        word2vec_similarity = calculate_pairwise_similarity(
            word2vec_tokens,
            word2vec_embeddings,
            "word2vec"
        )

        print("\n[✓] Word2Vec experiment completed.")

        # ====================================================
        # STEP 6
        # ====================================================

        print("\n")
        print("#" * 78)
        print("# STEP 6 — CODE2VEC-INSPIRED EMBEDDING")
        print("#" * 78)

        code2vec_embeddings = generate_code2vec_embeddings(
            selected_tokens,
            combined_dataset,
            tokenizer
        )

        code2vec_similarity = calculate_pairwise_similarity(
            selected_tokens,
            code2vec_embeddings,
            "code2vec"
        )

        print("\n[✓] Code2Vec-inspired experiment completed.")

        # ====================================================
        # STEP 7
        # ====================================================

        print("\n")
        print("#" * 78)
        print("# STEP 7 — COMPARE APPROACHES")
        print("#" * 78)

        compare_similarity_results()

        # ====================================================
        # COMPLETE
        # ====================================================

        elapsed = time.time() - start_time

        print("\n")
        print("=" * 78)
        print("                     LAB 4 COMPLETED")
        print("=" * 78)

        print(f"\nExecution time: {elapsed:.2f} seconds")

        print("\nCheck the output/ directory for:")
        print("  • selected tokens")
        print("  • custom embeddings")
        print("  • similarity results")
        print("  • Word2Vec model")
        print("  • Code2Vec-inspired embeddings")
        print("  • comparison results")

    except Exception as error:

        print("\n")
        print("=" * 78)
        print("[ERROR] LAB 4 PIPELINE FAILED")
        print("=" * 78)

        print(f"\nReason:")
        print(f"{type(error).__name__}: {error}")

        print(
            "\nThe pipeline stopped so the failing "
            "stage can be identified."
        )

        raise


if __name__ == "__main__":
    main()
