import logging

logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%d/%m/%Y %I:%M:%S %p"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from nebullvm.api.frontend.torch import optimize_torch_model  # noqa F401
from nebullvm.api.frontend.tf import optimize_tf_model  # noqa F401
from nebullvm.api.frontend.onnx import optimize_onnx_model  # noqa F401

__all__ = [k for k in globals().keys() if not k.startswith("_")]
