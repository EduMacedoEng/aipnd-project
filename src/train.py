import argparse
from utils import utils  # Adjusted import statement 

def main():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint.')
    
    parser.add_argument('--data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='./', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Architecture of the model to use')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    
    args = parser.parse_args()
    
    # Call the train function
    utils.train_model(args.data_dir, args.save_dir, args.arch,
                                args.learning_rate, args.hidden_units,
                                args.epochs, args.gpu)

if __name__ == '__main__':
    main()
    