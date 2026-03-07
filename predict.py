import argparse

import torch

from noise2noise import Model


def main():
    parser = argparse.ArgumentParser(description="Denoise images using a trained Noise2Noise model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model weights")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input data pickle file (noisy_imgs, clean_imgs)"
    )
    parser.add_argument("--output", type=str, default=None, help="Path to save denoised images pickle file")
    args = parser.parse_args()

    noisy_imgs, clean_imgs = torch.load(args.input, weights_only=True)

    model = Model()
    model.load_pretrained_model(args.model)

    denoised = model.predict(noisy_imgs)

    if args.output:
        torch.save(denoised, args.output)
        print(f"Denoised images saved to {args.output}")

    # Print PSNR
    mse = torch.mean((denoised.cpu() / 255 - clean_imgs.float().cpu() / 255) ** 2)
    psnr = -10 * torch.log10(mse + 1e-8)
    print(f"PSNR: {psnr.item():.2f} dB")


if __name__ == "__main__":
    main()
