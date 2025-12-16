"""Simple demo runner for the segmentation infer_images function.

Usage:
    python run_infer_demo.py <model_name> <image_path1> [image_path2 ...]

This loads the specified model and runs inference on the provided image paths,
writing masks to backend/output/<model_name>/ and printing results.
"""
import sys
from eval import infer_images

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_infer_demo.py <model_name> <image_path1> [image_path2 ...]")
        sys.exit(1)

    model_name = sys.argv[1]
    paths = sys.argv[2:]

    print(f"Running demo inference using model={model_name} on {len(paths)} images")
    outputs = infer_images(model_name, paths, threshold=0.5, out_dir=model_name, save_features=False)
    print("Outputs:")
    for p in outputs:
        print("  ", p)

if __name__ == '__main__':
    main()
