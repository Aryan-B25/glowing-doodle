"""
Main script to run the word embedding analysis for sentiment tasks
"""
from word_embedding_analysis import WordEmbeddingAnalysis


def main():
    # Initialize analyzer
    analyzer = WordEmbeddingAnalysis(data_dir='data')

    # Run analysis pipeline
    print("Starting word embedding analysis pipeline...")
    results = analyzer.run_pipeline()

    # Compare models
    comparison_df = analyzer.compare_models(
        results['similarity_results'],
        results['similarity_scores']
    )

    print("\nAnalysis complete. Results saved to output files.")
    print("Please check the current directory for visualization images and CSV outputs.")


if __name__ == "__main__":
    main()