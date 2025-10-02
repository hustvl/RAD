import torch.nn as nn

def set_bn_eval(module: nn.Module, exclude_names=None, prefix: str = ""):
    """
    Recursively set all normalization layers to evaluation mode, except for specified submodules.

    This is useful during RL fine-tuning to freeze normalization statistics
    (e.g., BatchNorm, LayerNorm) in parts of the model such as the planning head,
    ensuring stable training dynamics.

    Args:
        module (nn.Module): Root module to apply the setting.
        exclude_names (list[str], optional): List of substrings. If a module's
            full name contains any of these, it will be skipped.
            Defaults to common value function decoder branches.
        prefix (str, optional): Internal argument used to track module names
            during recursion. Users should not need to set this manually.
    """
    # If this module is a normalization layer and is not excluded, set it to eval mode.
    if isinstance(module, (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.LayerNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
        nn.GroupNorm,
    )):
        module.eval()
        # You can uncomment the line below for debugging.
        # print(f"[Norm Frozen] {prefix} ({type(module).__name__})")

    # Recurse through child modules unless the current module is excluded.
    if isinstance(module, nn.Module):
        if any(excluded in prefix for excluded in exclude_names):
            return
        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            set_bn_eval(child, exclude_names, child_prefix)
