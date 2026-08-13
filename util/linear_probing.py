def get_linear_probing_freeze(model):
    """
    Get the list of layers to freeze for linear probing on a YOLOE model.
    Linear probing fine-tunes only the classification branch while freezing the rest of the model.

    Args:
        model: YOLOE model instance

    Returns:
        list: List of layer indices/names to freeze as strings
    """
    # Identify the head layer index
    head_index = len(model.model.model) - 1

    # Freeze all backbone and neck layers (i.e. everything before the head)
    freeze = [str(i) for i in range(0, head_index)]

    # Freeze parts of the head, keeping only the classification branch trainable
    for name, child in model.model.model[-1].named_children():
        if "cv3" not in name:
            freeze.append(f"{head_index}.{name}")

    # Freeze detection branch components (bbox regression parts of cv3)
    freeze.extend(
        [
            f"{head_index}.cv3.0.0",
            f"{head_index}.cv3.0.1",
            f"{head_index}.cv3.1.0",
            f"{head_index}.cv3.1.1",
            f"{head_index}.cv3.2.0",
            f"{head_index}.cv3.2.1",
        ]
    )

    return freeze


def get_backbone_neck_freeze(model):
    """
    Get the list of layers to freeze for backbone and neck freezing on a YOLOE model.

    This function freezes all layers except the head, allowing fine-tuning of the head
    while keeping the backbone and neck fixed.

    Args:
        model: YOLOE model instance

    Returns:
        list: List of layer indices to freeze as strings
    """
    # Identify the head layer index
    head_index = len(model.model.model) - 1

    # Freeze all backbone and neck layers (i.e. everything before the head)
    freeze = [str(i) for i in range(0, head_index)]

    return freeze


def get_backbone_freeze(model):
    """
    Get the list of layers to freeze for backbone freezing on a YOLOE model.

    This function freezes all layers up to the SPPF layer, allowing fine-tuning of the neck and head while keeping the backbone fixed.

    Args:
        model: YOLOE model instance

    Returns:
        list: List of layer indices to freeze as strings
    """
    # Find the index of the SPPF layer
    SPPF_index = None
    for i in range(len(model.model.model)):
        if model.model.model[i]._get_name() == "SPPF":
            SPPF_index = i
            break

    assert SPPF_index is not None, "SPPF layer not found in the model"

    # Freeze all layers before SPPF
    freeze = [str(i) for i in range(0, SPPF_index)]

    return freeze
