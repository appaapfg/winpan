print('==============1==============', flush=True)
from .detect import Detector  # noqa: F401
print('==============2==============', flush=True)
from .hf_to_coco import HFdetections2COCOdetections  # noqa: F401
print('==============3==============', flush=True)

# from .segment import Segmenter

# __all__ = ["Detector", "Segmenter", "HFdetections2COCOdetections"]
