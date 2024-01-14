import argparse
from utils import utils  # Adjusted import statement 

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image with the probability of that name.')

    parser.add_argument('--image_path', type=str, help='Path to image')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--top_k', type=int, default=3, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='../cat_to_name.json', help='Use a mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')

    args = parser.parse_args()
    
    print("=========================================")
    print("          Predict Configuration")
    print("=========================================")
    print("- Image path:\t\t", args.image_path)
    print("- Checkpoint path:\t", args.checkpoint)
    print("- Top K:\t\t", args.top_k)
    print("- Category filepath:\t", args.category_names)
    print("- GPU Enabled:\t\t", args.gpu)
    print("=========================================\n")

    # Call the predict function
    utils.predict_image(args.image_path, args.checkpoint,
                        args.top_k, args.category_names,
                        args.gpu)

if __name__ == '__main__':
    main()