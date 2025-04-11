import numpy as np
import argparse

def save_embeddings_to_tsv(input_npy_path, output_tsv_path):
    embeddings = np.load(input_npy_path)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D array of shape [samples, features], got {embeddings.shape}")

    with open(output_tsv_path, 'w', encoding='utf-8') as f:
        for row in embeddings:
            line = '\t'.join([str(x) for x in row])
            f.write(line + '\n')

    print(f"Saved {embeddings.shape[0]} embeddings to {output_tsv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .npy embeddings to .tsv format for TensorFlow Projector.")
    parser.add_argument("input_npy", help="Path to input .npy file")
    parser.add_argument("output_tsv", help="Path to output .tsv file")
    args = parser.parse_args()

    save_embeddings_to_tsv(args.input_npy, args.output_tsv)
