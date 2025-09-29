import os
import warnings

def ignore_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r".*image Python extension.*",   # regex match
        category=UserWarning,
        module=r"torchvision\.io\.image"         # regex match
    )
    # Also blanket-ignore any leftover UserWarnings from torchvision
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision(\.|$)")

    os.environ.setdefault("MPLBACKEND", "Agg")
