import json
import shutil
from abc import ABC
from pathlib import Path
from typing import Dict, Union, Type, Generator, Tuple, List, Optional

import numpy as np
from loguru import logger

from nebullvm.config import OPENVINO_FILENAMES
from nebullvm.core.models import Device, ModelParams, DeepLearningFramework
from nebullvm.operations.inference_learners.base import (
    BaseInferenceLearner,
    LearnerMetadata,
    PytorchBaseInferenceLearner,
    TensorflowBaseInferenceLearner,
    NumpyBaseInferenceLearner,
)
from nebullvm.optional_modules.openvino import (
    Core,
    Model,
    CompiledModel,
)
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import torch
from nebullvm.tools.data import DataManager
from nebullvm.tools.transformations import MultiStageTransformation


class OpenVinoInferenceLearner(BaseInferenceLearner, ABC):
    """Model optimized using OpenVINO.

    The class cannot be directly instantiated, but implements all the core
    methods needed for using OpenVINO at inference time.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        exec_network (any): The graph executor. This is the
            central component in the OpenVino optimized model execution.
        input_keys (List): Keys associated to the inputs.
        output_keys (List): Keys associated to the outputs.
        description_file (str): File containing a description of the optimized
            model.
        weights_file (str): File containing the model weights.
    """

    MODEL_NAME = "model.bin"
    name = "OpenVINO"

    def __init__(
        self,
        compiled_model: CompiledModel,
        input_keys: List,
        output_keys: List,
        description_file: str,
        weights_file: str,
        device: Device,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.compiled_model = compiled_model
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.device = device
        self.description_file = self._store_file(description_file)
        self.weights_file = self._store_file(weights_file)

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        """Load the model.

        Args:
            path (Path or str): Path to the directory where the model is
                stored.
            kwargs (Dict): Dictionary of additional arguments for the
                `from_model_name` class method.

        Returns:
            OpenVinoInferenceLearner: The optimized model.
        """
        path = Path(path)

        with open(path / OPENVINO_FILENAMES["metadata"], "r") as fin:
            metadata = json.load(fin)
        metadata.update(kwargs)
        metadata["network_parameters"] = ModelParams(
            **metadata["network_parameters"]
        )
        input_tfms = metadata.get("input_tfms")
        if input_tfms is not None:
            metadata["input_tfms"] = MultiStageTransformation.from_dict(
                input_tfms
            )

        model_name = str(path / OPENVINO_FILENAMES["description_file"])
        model_weights = str(path / OPENVINO_FILENAMES["weights"])
        metadata["device"] = Device.from_str(metadata["device"])
        return cls.from_model_name(
            model_name=model_name, model_weights=model_weights, **metadata
        )

    def get_size(self):
        return len(self.compiled_model.export_model())

    def free_gpu_memory(self):
        raise NotImplementedError("OpenVino does not support GPU inference.")

    @classmethod
    def from_model_name(
        cls,
        network_parameters: ModelParams,
        model_name: str,
        model_weights: str,
        device: Device,
        input_tfms: MultiStageTransformation = None,
        input_data: DataManager = None,
        **kwargs,
    ):
        """Build the optimized model from the network description and its
        weights.

        Args:
            network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
            model_name (str): File containing a description of the optimized
                model.
            model_weights (str): File containing the model weights.
            device (Device): Device used to run the model.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction.
            input_data (DataManager, optional): User defined data.
        """
        if len(kwargs) > 0:
            logger.warning(f"Found extra parameters: {kwargs}")

        core = Core()
        model = core.read_model(model=model_name, weights=model_weights)

        dynamic_shape = cls._get_dynamic_shape(model, network_parameters)

        if dynamic_shape is not None:
            model.reshape(dynamic_shape)

        compiled_model = core.compile_model(model=model, device_name="CPU")

        input_keys = list(
            map(lambda obj: obj.get_any_name(), compiled_model.inputs)
        )
        output_keys = list(
            map(lambda obj: obj.get_any_name(), compiled_model.outputs)
        )

        return cls(
            compiled_model,
            input_keys,
            output_keys,
            input_tfms=input_tfms,
            network_parameters=network_parameters,
            description_file=model_name,
            weights_file=model_weights,
            input_data=input_data,
            device=device,
        )

    @staticmethod
    def _get_dynamic_shape(
        model: Model, network_parameters: ModelParams
    ) -> Optional[Dict[str, Tuple[int]]]:
        if network_parameters.dynamic_info is None:
            return None

        input_names = [
            list(model_input.names)[0] for model_input in model.inputs
        ]
        input_shapes = [
            input_info.size for input_info in network_parameters.input_infos
        ]
        dynamic_shapes = []

        assert len(input_shapes) == len(
            network_parameters.dynamic_info.inputs
        ), (
            f"Number of inputs defined in dynamic info "
            f"({len(input_shapes)}) is different from the one "
            f"expected from the model "
            f"({len(network_parameters.dynamic_info.inputs)})."
        )

        for input_shape, dynamic_shape_dict in zip(
            input_shapes, network_parameters.dynamic_info.inputs
        ):
            input_shape = list(input_shape)
            for key in dynamic_shape_dict.keys():
                input_shape[int(key)] = -1
            dynamic_shapes.append(tuple(input_shape))

        dynamic_shape_dict = {
            k: v for k, v in zip(input_names, dynamic_shapes)
        }
        return dynamic_shape_dict

    def _get_metadata(self, **kwargs) -> LearnerMetadata:
        # metadata = {
        #     key: self.__dict__[key] for key in ("input_keys", "output_keys")
        # }
        metadata = {}
        metadata.update(kwargs)
        return LearnerMetadata.from_model(self, **metadata)

    def save(self, path: Union[str, Path], **kwargs):
        """Save the model.

        Args:
            path (Path or str): Path to the directory where the model will
                be stored.
            kwargs (Dict): Dictionary of key-value pairs that will be saved in
                the model metadata file.
        """
        path = Path(path)
        path.mkdir(exist_ok=True)
        metadata = self._get_metadata(**kwargs)

        metadata.save(path)

        shutil.copy(
            self.description_file,
            path / OPENVINO_FILENAMES["description_file"],
        )
        shutil.copy(self.weights_file, path / OPENVINO_FILENAMES["weights"])

    def _predict_array(
        self,
        input_arrays: Generator[np.ndarray, None, None],
    ) -> Generator[np.ndarray, None, None]:

        results = self.compiled_model(
            inputs={
                input_key: input_array
                for input_key, input_array in zip(
                    self.input_keys, input_arrays
                )
            },
            shared_memory=True,  # always enabled
        )

        return (results[output_key] for output_key in self.output_keys)


class PytorchOpenVinoInferenceLearner(
    OpenVinoInferenceLearner, PytorchBaseInferenceLearner
):
    """Model optimized using ApacheTVM with a Pytorch interface.

    This class can be used exactly in the same way as a pytorch Module object.
    At prediction time it takes as input pytorch tensors given as positional
    arguments.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        exec_network (any): The graph executor. This is the
            central component in the OpenVino optimized model execution.
        input_keys (List): Keys associated to the inputs.
        output_keys (List): Keys associated to the outputs.
        description_file (str): File containing a description of the optimized
            model.
        weights_file (str): File containing the model weights.
    """

    def run(self, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Predict on the input tensors.

        Note that the input tensors must be on the same batch. If a sequence
        of tensors is given when the model is expecting a single input tensor
        (with batch size >= 1) an error is raised.

        Args:
            input_tensors (Tuple[Tensor]): Input tensors belonging to the same
                batch. The tensors are expected having dimensions
                (batch_size, dim1, dim2, ...).

        Returns:
            Tuple[Tensor]: Output tensors. Note that the output tensors does
                not correspond to the prediction on the input tensors with a
                1 to 1 mapping. In fact the output tensors are produced as the
                multiple-output of the model given a (multi-) tensor input.
        """
        input_arrays = (
            input_tensor.cpu().detach().numpy()
            for input_tensor in input_tensors
        )
        output_arrays = self._predict_array(input_arrays)
        return tuple(
            torch.from_numpy(output_array) for output_array in output_arrays
        )


class TensorflowOpenVinoInferenceLearner(
    OpenVinoInferenceLearner, TensorflowBaseInferenceLearner
):
    """Model optimized using ApacheTVM with a tensorflow interface.

    This class can be used exactly in the same way as a tf.Module or
    keras.Model object.
    At prediction time it takes as input tensorflow tensors given as positional
    arguments.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        exec_network (any): The graph executor. This is the
            central component in the OpenVino optimized model execution.
        input_keys (List): Keys associated to the inputs.
        output_keys (List): Keys associated to the outputs.
        description_file (str): File containing a description of the optimized
            model.
        weights_file (str): File containing the model weights.
    """

    def run(self, *input_tensors: tf.Tensor) -> Tuple[tf.Tensor, ...]:
        """Predict on the input tensors.

        Note that the input tensors must be on the same batch. If a sequence
        of tensors is given when the model is expecting a single input tensor
        (with batch size >= 1) an error is raised.

        Args:
            input_tensors (Tuple[Tensor]): Input tensors belonging to the same
                batch. The tensors are expected having dimensions
                (batch_size, dim1, dim2, ...).

        Returns:
            Tuple[Tensor]: Output tensors. Note that the output tensors does
                not correspond to the prediction on the input tensors with a
                1 to 1 mapping. In fact the output tensors are produced as the
                multiple-output of the model given a (multi-) tensor input.
        """
        input_arrays = (input_tensor.numpy() for input_tensor in input_tensors)
        output_arrays = self._predict_array(input_arrays)
        # noinspection PyTypeChecker
        return tuple(
            tf.convert_to_tensor(output_array)
            for output_array in output_arrays
        )


class NumpyOpenVinoInferenceLearner(
    OpenVinoInferenceLearner, NumpyBaseInferenceLearner
):
    """Model optimized using ApacheTVM with a numpy interface.

    This class can be used exactly in the same way as a sklearn or
    numpy-based model.
    At prediction time it takes as input numpy arrays given as positional
    arguments.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        exec_network (any): The graph executor. This is the
            central component in the OpenVino optimized model execution.
        input_keys (List): Keys associated to the inputs.
        output_keys (List): Keys associated to the outputs.
        description_file (str): File containing a description of the optimized
            model.
        weights_file (str): File containing the model weights.
    """

    def run(self, *input_tensors: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Predict on the input tensors.

        Note that the input tensors must be on the same batch. If a sequence
        of tensors is given when the model is expecting a single input tensor
        (with batch size >= 1) an error is raised.

        Args:
            input_tensors (Tuple[np.ndarray]): Input tensors belonging to
                the same batch. The tensors are expected having dimensions
                (batch_size, dim1, dim2, ...).

        Returns:
            Tuple[np.ndarray]: Output tensors. Note that the output tensors
                does not correspond to the prediction on the input tensors
                with a 1 to 1 mapping. In fact the output tensors are produced
                as the multiple-output of the model given a (multi-) tensor
                input.
        """
        input_arrays = (input_tensor for input_tensor in input_tensors)
        output_arrays = self._predict_array(input_arrays)
        return tuple(output_arrays)


OPENVINO_INFERENCE_LEARNERS: Dict[
    DeepLearningFramework, Type[OpenVinoInferenceLearner]
] = {
    DeepLearningFramework.PYTORCH: PytorchOpenVinoInferenceLearner,
    DeepLearningFramework.TENSORFLOW: TensorflowOpenVinoInferenceLearner,
    DeepLearningFramework.NUMPY: NumpyOpenVinoInferenceLearner,
}
