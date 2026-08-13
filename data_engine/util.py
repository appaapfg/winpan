import os
from contextlib import redirect_stdout

from pycocotools.coco import COCO


def load_coco(input_dataset: str | COCO | dict, silent: bool = False) -> COCO:
    """Loads a COCO dataset from various input types into a COCO object.

    This utility function provides a flexible way to initialize a `pycocotools.coco.COCO`
    object. It can handle a direct file path to a COCO JSON annotation file,
    an already-existing COCO object, or a Python dictionary that conforms to the
    COCO data structure. This allows for seamless integration whether the dataset
    is read from a file or is constructed in-memory.

    Args:
        input_dataset: The dataset to load. This can be one of three types:
            - str: The file path to the COCO format JSON annotation file.
            - pycocotools.coco.COCO: An existing COCO object. The function
              will simply return this object.
            - dict: A dictionary containing COCO-structured data. The function
              will manually initialize a COCO object with this dictionary.
        silent: If True, suppresses the output from pycocotools.

    Returns:
        A `pycocotools.coco.COCO` object initialized with the provided dataset.
        This object can be used to access the COCO API for retrieving annotations,
        image info, etc.

    Raises:
        AssertionError: If `input_dataset` is a string path and the file
            does not exist.
        ValueError: If `input_dataset` is not a string, a COCO object,
            or a dictionary.

    Examples:
        >>> # Example 1: Loading from a JSON file path
        >>> json_file_path = 'path/to/your/annotations.json'
        >>> # create_dummy_coco_file(json_file_path)  # Helper to create a file for demo
        >>> # coco_from_path = load_coco(json_file_path, silent=True)
        >>> # print(coco_from_path.getCatIds())

        >>> # Example 2: Passing an existing COCO object
        >>> # existing_coco = COCO(json_file_path)
        >>> # coco_from_object = load_coco(existing_coco)
        >>> # assert id(existing_coco) == id(coco_from_object)

        >>> # Example 3: Loading from a Python dictionary
        >>> coco_dict = {
        ...     'images': [{'id': 1, 'file_name': 'image1.jpg', 'height': 480, 'width': 640}],
        ...     'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 50, 50]}],
        ...     'categories': [{'id': 1, 'name': 'cat', 'supercategory': 'animal'}]
        ... }
        >>> coco_from_dict = load_coco(coco_dict, silent=True)
        >>> # img_ids = coco_from_dict.getImgIds()
        >>> # print(f"Image IDs from dictionary: {img_ids}")
    """

    def _load():
        if isinstance(input_dataset, str):
            assert os.path.isfile(input_dataset), f"COCO JSON not found: {input_dataset}"
            return COCO(input_dataset)
        elif isinstance(input_dataset, COCO):
            return input_dataset
        elif isinstance(input_dataset, dict):
            coco = COCO()
            coco.dataset = input_dataset
            coco.createIndex()
            return coco
        else:
            raise ValueError("coco_input must be a str path or a COCO object")

    if silent:
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull):
                return _load()
    else:
        return _load()
