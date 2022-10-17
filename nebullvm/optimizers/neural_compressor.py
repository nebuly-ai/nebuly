from typing import Optional, Callable, Any

import torch

from nebullvm.base import ModelParams, DeepLearningFramework, QuantizationType
from nebullvm.inference_learners.neural_compressor import (
    NEURAL_COMPRESSOR_INFERENCE_LEARNERS,
    NeuralCompressorInferenceLearner,
)
from nebullvm.optimizers import BaseOptimizer
from nebullvm.optimizers.quantization.neural_compressor import (
    quantize_neural_compressor,
)
from nebullvm.optimizers.quantization.utils import (
    check_quantization,
    check_precision,
)
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager
from nebullvm.utils.onnx import (
    convert_to_target_framework,
)


class NeuralCompressorOptimizer(BaseOptimizer):
    def optimize(
        self,
        model: torch.nn.Module,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
        model_outputs: Any = None,
    ) -> Optional[NeuralCompressorInferenceLearner]:
        """Optimize the input model using Intel Neural Compressor Quantization.

        Args:
            model (torch.nn.Module): The pytorch model. For avoiding un-wanted
                modifications to the original model, it will be copied in the
                method.
            output_library (DeepLearningFramework): Output framework. At the
                current stage just PYTORCH is supported.
            model_params (ModelParams): Model parameters.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction. Default: None.
            metric_drop_ths (float, optional): Threshold for the accepted drop
                in terms of precision. Any optimized model with an higher drop
                will be ignored. Default: None.
            quantization_type (QuantizationType, optional): The desired
                quantization algorithm to be used. Default: None.
            metric (Callable, optional): If given it should
                compute the difference between the quantized and the normal
                prediction. Default: None.
            input_data (DataManager, optional): User defined data.
                Default: None.
            model_outputs (Any, optional): Outputs computed by the original
                model. Default: None.

        Returns:
            NeuralCompressorInferenceLearner: Model optimized for inference.
        """
        self._log(
            f"Optimizing with {self.__class__.__name__} and "
            f"q_type: {quantization_type}."
        )

        assert output_library is DeepLearningFramework.PYTORCH, (
            "Other APIs than the Pytorch one are not supported "
            "for the Neural Compressor yet."
        )
        check_quantization(quantization_type, metric_drop_ths)

        # At the moment only static quantization is supported
        if quantization_type is not QuantizationType.STATIC:
            return None

        if metric_drop_ths is not None:
            input_data_torch, ys = input_data.get_numpy_list(300, with_ys=True)
            input_data_torch = [
                tuple(
                    convert_to_target_framework(t, output_library)
                    for t in data_tuple
                )
                for data_tuple in input_data_torch
            ]
            output_data_torch = model_outputs
            model_quant = quantize_neural_compressor(
                model, quantization_type, input_data
            )

            learner = NEURAL_COMPRESSOR_INFERENCE_LEARNERS[output_library](
                input_tfms=input_tfms,
                network_parameters=model_params,
                model=model_quant,
            )

            if metric_drop_ths is not None:
                is_valid = check_precision(
                    learner,
                    input_data_torch,
                    output_data_torch,
                    metric_drop_ths,
                    metric_func=metric,
                    ys=ys,
                )
                if not is_valid:
                    return None
            return learner
        else:
            return None