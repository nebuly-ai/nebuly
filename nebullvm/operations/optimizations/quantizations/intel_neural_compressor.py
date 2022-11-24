from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from nebullvm.base import QuantizationType
from nebullvm.compressors.intel import INCDataset
from nebullvm.operations.optimizations.quantizations.base import Quantizer
from nebullvm.optional_modules.neural_compressor import (
    MixedPrecision,
    Quantization,
)
from nebullvm.optional_modules.torch import DataLoader, Module, GraphModule
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.transformations.precision_tfms import HalfPrecisionTransformation
from nebullvm.utils.data import DataManager


def _prepare_quantization_config(model: Any, tmp_dir: str, approach: str):
    config = {
        "model": {
            "name": model.__class__.__name__,
            "framework": "pytorch_fx",
        },
        "quantization": {"approach": approach},
        "evaluation": {"accuracy": {"metric": {"topk": 1}}},
        "tuning": {
            "accuracy_criterion": {"relative": 0.01},
        },
    }

    path_file = Path(tmp_dir) / "temp_qt.yaml"
    with open(path_file, "w") as f:
        yaml.dump(config, f)

    return path_file


def _prepare_mixed_precision_config(model: Any, tmp_dir: str):
    config = {
        "model": {
            "name": model.__class__.__name__,
            "framework": "pytorch_fx",
        },
        "mixed_precision": {"precisions": "bf16"},
        "evaluation": {"accuracy": {"metric": {"topk": 1}}},
        "tuning": {
            "accuracy_criterion": {"relative": 0.01},
        },
    }

    path_file = Path(tmp_dir) / "temp_mp.yaml"
    with open(path_file, "w") as f:
        yaml.dump(config, f)

    return path_file


def _get_dataloader(input_data: DataManager):
    bs = input_data[0][0][0].shape[0]
    ds = INCDataset(input_data)
    dl = DataLoader(ds, bs)
    return dl


def _quantize_static(model: Module, input_data: DataManager) -> GraphModule:
    with TemporaryDirectory() as tmp_dir:
        config_file_qt = _prepare_quantization_config(
            model, tmp_dir, "post_training_static_quant"
        )
        quantizer = Quantization(str(config_file_qt))
        quantizer.model = model
        quantizer.calib_dataloader = _get_dataloader(input_data)
        quantizer.eval_dataloader = _get_dataloader(input_data)
        compressed_model = quantizer()

    return compressed_model


def _quantize_dynamic(model: Module) -> GraphModule:
    with TemporaryDirectory() as tmp_dir:
        config_file_qt = _prepare_quantization_config(
            model, tmp_dir, "post_training_dynamic_quant"
        )
        quantizer = Quantization(str(config_file_qt))
        quantizer.model = model
        compressed_model = quantizer()

    return compressed_model


def _mixed_precision(
    model: Module, input_tfms: MultiStageTransformation
) -> GraphModule:
    with TemporaryDirectory() as tmp_dir:
        config_file_qt = _prepare_mixed_precision_config(model, tmp_dir)
        converter = MixedPrecision(str(config_file_qt))
        converter.model = model
        compressed_model = converter()

    input_tfms.append(HalfPrecisionTransformation())

    return compressed_model


class IntelNeuralCompressorQuantizer(Quantizer):
    def execute(
        self,
        model: Module,
        quantization_type: QuantizationType,
        input_tfms: MultiStageTransformation,
        input_data: DataManager,
    ):
        if quantization_type is QuantizationType.STATIC:
            self.quantized_model = _quantize_static(model, input_data)
        elif quantization_type is QuantizationType.DYNAMIC:
            self.quantized_model = _quantize_dynamic(model)
        elif quantization_type is QuantizationType.HALF:
            self.quantized_model = _mixed_precision(model, input_tfms)
        else:
            raise ValueError(
                f"Quantization type {quantization_type} is not "
                f"supported by Intel Neural Compressor"
            )
