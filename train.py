import argparse

import torch

from noise2noise import Model


def main():
    parser = argparse.ArgumentParser(description="Train a Noise2Noise denoising model")
    parser.add_argument("--train-data", type=str, required=True, help="Path to training data pickle file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of data loading workers (default: 2)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--output", type=str, default="model_pytorch.pth", help="Output path for trained model")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained model to resume from")
    args = parser.parse_args()

    noisy_imgs_1, noisy_imgs_2 = torch.load(args.train_data, weights_only=True)

    model = Model(lr=args.lr)
    if args.pretrained:
        model.load_pretrained_model(args.pretrained)

    model.train(
        noisy_imgs_1, noisy_imgs_2, num_epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers
    )
    model.save(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
