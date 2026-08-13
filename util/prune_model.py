import argparse

import torch
import torch.nn.utils.prune as prune


def prune_conv2d_layers(model, amount=0.2, method="l1_unstructured"):
    """
    Prune Conv2d layers in a PyTorch model using torch.nn.utils.prune.

    Args:
        model (torch.nn.Module): The model to prune.
        amount (float): The fraction of weights to prune (0.0 to 1.0).
        method (str): The pruning method to use. Currently supports 'l1_unstructured'.

    Returns:
        torch.nn.Module: The pruned model (modified in-place).
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if method == "l1_unstructured":
                prune.l1_unstructured(module, name="weight", amount=amount)
            else:
                raise ValueError(f"Unsupported pruning method: {method}")
            prune.remove(module, "weight")

    return model


def main():
    parser = argparse.ArgumentParser(description="Prune Conv2d layers in a PyTorch model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the PyTorch model file (.pt or .pth)")
    parser.add_argument("--amount", type=float, default=0.2, help="Fraction of weights to prune (0.0 to 1.0)")
    parser.add_argument("--method", type=str, default="l1_unstructured", help="Pruning method")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the pruned model")

    args = parser.parse_args()

    model = torch.load(args.model_path)
    prune_conv2d_layers(model, amount=args.amount, method=args.method)

    if args.output_path:
        torch.save(model, args.output_path)
        print(f"Pruned model saved to {args.output_path}")
    else:
        print("Model pruned in-place. Provide --output_path to save.")


if __name__ == "__main__":
    main()
