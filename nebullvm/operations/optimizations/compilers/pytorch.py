from typing import Callable, Any, Union

from nebullvm.base import Device, QuantizationType
from nebullvm.config import QUANTIZATION_DATA_NUM
from nebullvm.operations.optimizations.compilers.base import Compiler
from nebullvm.operations.optimizations.quantizations.pytorch import (
    PytorchQuantizer,
)
from nebullvm.operations.optimizations.quantizations.utils import (
    check_quantization,
)
from nebullvm.optional_modules.torch import (
    torch,
    Module,
    ScriptModule,
    GraphModule,
    symbolic_trace,
)
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager


class PytorchBackendCompiler(Compiler):
    supported_ops = {
        "cpu": [None, QuantizationType.STATIC, QuantizationType.DYNAMIC],
        "gpu": [
            None,
            QuantizationType.STATIC,
            QuantizationType.HALF,
            QuantizationType.DYNAMIC,
        ],
    }

    def __init__(self):
        super().__init__()
        self.quantization_op = PytorchQuantizer()

    def execute(
        self,
        model: Module,
        input_data: DataManager,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        model_outputs: Any = None,
    ):
        """Optimize the input model using pytorch built-in techniques.

        Args:
            model (torch.nn.Module): The pytorch model. For avoiding un-wanted
                modifications to the original model, it will be copied in the
                method.
            input_data (DataManager): User defined data. Default: None.
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
            model_outputs (Any, optional): Outputs computed by the original
                model. Default: None.

        Returns:
            PytorchBackendInferenceLearner: Model optimized for inference.
        """

        if quantization_type not in self.supported_ops[self.device.value]:
            self.compiled_model = None
            return

        self.logger.info(
            f"Optimizing with {self.__class__.__name__} and "
            f"q_type: {quantization_type}."
        )

        check_quantization(quantization_type, metric_drop_ths)
        train_input_data = input_data.get_split("train").get_list(
            QUANTIZATION_DATA_NUM
        )

        if quantization_type is not None:
            self.quantization_op.to(self.device).execute(
                model, quantization_type, input_tfms, train_input_data
            )
            model = self.quantization_op.get_result()

        self.compiled_model = self.compile_model(model, input_data)

    def compile_model(
        self,
        model: Union[Module, GraphModule],
        input_data: DataManager,
    ) -> ScriptModule:
        if self.device is Device.GPU:
            input_data = [t.cuda() for t in input_data]

        if not isinstance(model, torch.fx.GraphModule):
            model.eval()
            try:
                model_scripted = symbolic_trace(model)
                model_scripted = torch.jit.script(model_scripted)
            except Exception:
                try:
                    model_scripted = torch.jit.script(model)
                except Exception:
                    model_scripted = torch.jit.trace(model, tuple(input_data))
        else:
            model_scripted = torch.jit.script(model)

        return model_scripted
