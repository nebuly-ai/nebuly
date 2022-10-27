import logging

logger = logging.getLogger(__name__)

try:
    import torch_blade
except ImportError:
    logger.warning(
        "torch_blade module is not installed on this platform. "
        "Please install it if you want to include it in the "
        "optimization pipeline."
    )

    torch_blade = object
