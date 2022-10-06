import os
import subprocess
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Any

import numpy as np
import torch

from nebullvm.base import DeepLearningFramework, ModelParams, QuantizationType
from nebullvm.config import (
    NVIDIA_FILENAMES,
    NO_COMPILER_INSTALLATION,
    TORCH_TENSORRT_PRECISIONS,
)
from nebullvm.inference_learners.tensor_rt import (
    NVIDIA_INFERENCE_LEARNERS,
    NvidiaInferenceLearner,
    PytorchTensorRTInferenceLearner,
)
from nebullvm.optimizers.base import (
    BaseOptimizer,
)
from nebullvm.optimizers.quantization.tensor_rt import TensorRTCalibrator
from nebullvm.optimizers.quantization.utils import (
    check_precision,
    check_quantization,
)
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager, PytorchDataset
from nebullvm.utils.general import check_module_version
from nebullvm.utils.onnx import (
    get_input_names,
    get_output_names,
)

if torch.cuda.is_available():
    try:
        import tensorrt as trt
    except ImportError:
        from nebullvm.installers.installers import install_tensor_rt

        if not NO_COMPILER_INSTALLATION:
            warnings.warn(
                "No TensorRT valid installation has been found. "
                "Trying to install it from source."
            )
            install_tensor_rt()
            import tensorrt as trt
        else:
            warnings.warn(
                "No TensorRT valid installation has been found. "
                "It won't be possible to use it in the following."
            )
    if check_module_version(torch, min_version="1.12.0"):
        try:
            import torch_tensorrt
        except ImportError:
            if not NO_COMPILER_INSTALLATION:
                from nebullvm.installers.installers import (
                    install_torch_tensor_rt,
                )

                warnings.warn(
                    "No Torch TensorRT valid installation has been found. "
                    "Trying to install it from source."
                )

                install_torch_tensor_rt()

                # Wrap import inside try/except because installation
                # may fail until wheel 1.2 will be officially out.
                try:
                    import torch_tensorrt
                except ImportError:
                    warnings.warn(
                        "Unable to install Torch TensorRT on this platform. "
                        "It won't be possible to use it in the following."
                    )
            else:
                warnings.warn(
                    "No Torch TensorRT valid installation has been found. "
                    "It won't be possible to use it in the following."
                )
    else:
        warnings.warn(
            "Torch-TensorRT can be installed only from Pytorch 1.12. "
            "Please update your Pytorch version."
        )


class TensorRTOptimizer(BaseOptimizer):
    """Class for compiling the AI models on Nvidia GPUs using TensorRT."""

    def _build_and_save_the_engine(
        self,
        engine_path: str,
        onnx_model_path: str,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        quantization_type: QuantizationType = None,
        input_data: List[Tuple[np.ndarray, ...]] = None,
    ):
        # -- Build phase --
        nvidia_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(nvidia_logger)
        # create network definition
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        # build the engine
        # TODO: setup config value for the class in a config file
        config = builder.create_builder_config()
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        except AttributeError:
            # The method set_memory_pool_limit is not available
            # until TensorRT Release 8.4.1
            warnings.warn(
                "Cannot call method set_memory_pool_limit for TensorRT."
                "Please update TensorRT version."
            )
        if quantization_type is QuantizationType.HALF:
            config.set_flag(trt.BuilderFlag.FP16)
        elif quantization_type is QuantizationType.STATIC:
            assert input_data is not None, (
                "You need to specify the calibration data for "
                "performing static quantization."
            )
            calibrator = TensorRTCalibrator(
                batch_size=model_params.batch_size,
                input_data=input_data,
            )
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calibrator
        elif quantization_type is QuantizationType.DYNAMIC:
            # onnx_model_path, _ = quantize_onnx(
            #     onnx_model_path, quantization_type, input_tfms, input_data
            # )
            config.set_flag(trt.BuilderFlag.INT8)
        # import the model
        parser = trt.OnnxParser(network, nvidia_logger)
        success = parser.parse_from_file(onnx_model_path)

        if not success:
            for idx in range(parser.num_errors):
                if self.logger is not None:
                    self.logger.debug(parser.get_error(idx))
            raise ValueError(
                f"Errors occurred while processing the "
                f"ONNX file at {onnx_model_path}"
            )

        if model_params.dynamic_info is not None:
            profile = builder.create_optimization_profile()
            for input_name, input_dynamic_info, input_info in zip(
                get_input_names(onnx_model_path),
                model_params.dynamic_info.inputs,
                model_params.input_infos,
            ):
                profile.set_shape(
                    input_name,
                    (
                        min(model_params.batch_size, 1)
                        if 0 in input_dynamic_info
                        else model_params.batch_size,
                        *(
                            shape
                            if i + 1 not in input_dynamic_info
                            else (input_info.min_sizes or {}).get(i + 1, 1)
                            for i, shape in enumerate(input_info.size)
                        ),
                    ),
                    (model_params.batch_size, *input_info.size),
                    (model_params.batch_size, *input_info.size),
                )
            config.add_optimization_profile(profile)
        serialized_engine = builder.build_serialized_network(network, config)
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

    def optimize(
        self,
        model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
        model_outputs: Any = None,
    ) -> Optional[NvidiaInferenceLearner]:
        """Optimize the input model with TensorRT.

        Args:
            model (str): Path to the saved onnx model.
            output_library (str): DL Framework the optimized model will be
                compatible with.
            model_params (ModelParams): Model parameters.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction.
            metric_drop_ths (float, optional): Threshold for the accepted drop
                in terms of precision. Any optimized model with an higher drop
                will be ignored.
            quantization_type (QuantizationType, optional): The desired
                quantization algorithm to be used.
            metric (Callable, optional): If given it should
                compute the difference between the quantized and the normal
                prediction.
            input_data (DataManager, optional): User defined data.
            model_outputs (Any): Outputs computed by the original model.

        Returns:
            TensorRTInferenceLearner: Model optimized with TensorRT. The model
                will have an interface in the DL library specified in
                `output_library`.
        """
        self._log(
            f"Optimizing with {self.__class__.__name__} and "
            f"q_type: {quantization_type}."
        )
        if not torch.cuda.is_available():
            raise SystemError(
                "You are trying to run an optimizer developed for NVidia gpus "
                "on a machine not connected to any GPU supporting CUDA."
            )

        # Simplify model, otherwise tensor RT won't work on gpt2 and some
        # other models.
        simplified_model = model + "_simplified"
        if not Path(simplified_model).is_file():
            cmd = [
                "onnxsim",
                model,
                model + "_simplified",
            ]
            subprocess.run(cmd)

        check_quantization(quantization_type, metric_drop_ths)
        engine_path = (
            Path(simplified_model).parent / NVIDIA_FILENAMES["engine"]
        )
        if (
            metric_drop_ths is not None
            and quantization_type is QuantizationType.STATIC
        ):
            input_data_onnx = input_data.get_numpy_list(300, with_ys=False)
        elif (
            metric_drop_ths is not None
            and quantization_type is QuantizationType.DYNAMIC
        ):
            return None  # Dynamic quantization is not supported on tensorRT
        else:
            input_data_onnx = None
        self._build_and_save_the_engine(
            engine_path=engine_path,
            onnx_model_path=simplified_model,
            model_params=model_params,
            input_tfms=input_tfms,
            quantization_type=quantization_type,
            input_data=input_data_onnx,
        )

        learner = NVIDIA_INFERENCE_LEARNERS[output_library].from_engine_path(
            input_tfms=input_tfms,
            network_parameters=model_params,
            engine_path=engine_path,
            input_names=get_input_names(simplified_model),
            output_names=get_output_names(simplified_model),
            input_data=list(input_data.get_list(1)[0])
            if input_data is not None
            else None,
        )
        if quantization_type is not None:
            inputs, ys = input_data.get_list(100, shuffle=True, with_ys=True)
            output_data = model_outputs
            is_valid = check_precision(
                learner,
                inputs,
                output_data,
                metric_drop_ths,
                metric_func=metric,
                ys=ys,
            )
            if not is_valid:
                return None
        return learner

    def optimize_from_torch(
        self,
        torch_model: torch.nn.Module,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
        model_outputs: Any = None,
    ) -> Optional[PytorchTensorRTInferenceLearner]:
        self._log(
            f"Optimizing with {self.__class__.__name__} and "
            f"q_type: {quantization_type}."
        )
        if not torch.cuda.is_available():
            raise SystemError(
                "You are trying to run an optimizer developed for NVidia gpus "
                "on a machine not connected to any GPU supporting CUDA."
            )

        check_quantization(quantization_type, metric_drop_ths)

        dtype = torch.float32

        if (
            metric_drop_ths is not None
            and quantization_type is QuantizationType.HALF
        ):
            dtype = torch.half
        elif (
            metric_drop_ths is not None
            and quantization_type is QuantizationType.STATIC
        ):
            dtype = torch.int8

            dataset = PytorchDataset(input_data)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                shuffle=False,
                num_workers=0,
            )

            calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
                dataloader,
                use_cache=False,
                algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,  # noqa E501
                device=torch.device("cuda:0"),
            )
        elif (
            metric_drop_ths is not None
            and quantization_type is QuantizationType.DYNAMIC
        ):
            return None  # Dynamic quantization is not supported on tensorRT

        # Convert int64 to int32 for transformers inputs
        input_tensors = [
            tensor.cuda()
            if tensor.dtype != torch.int64
            else tensor.to(torch.int32).cuda()
            for tensor in input_data.get_list(1)[0]
        ]

        torch_model.cuda().eval()

        try:
            torch.jit.script(torch_model)
            model = torch_model
        except Exception:
            model = torch.jit.trace(torch_model, input_tensors)

        trt_model = torch_tensorrt.compile(
            model,
            inputs=[
                torch_tensorrt.Input(
                    tensor.shape,
                    dtype=torch.half
                    if (
                        dtype == torch.half
                        and tensor.dtype not in [torch.int8, torch.int32]
                    )
                    else tensor.dtype,
                )
                for tensor in input_tensors
            ],
            enabled_precisions=TORCH_TENSORRT_PRECISIONS[str(dtype)],
            calibrator=calibrator if dtype == torch.int8 else None,
            workspace_size=1 << 30,
            device={
                "device_type": torch_tensorrt.DeviceType.GPU,
                "gpu_id": 0,
                "dla_core": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False,
            },
            truncate_long_and_double=True,
        )

        # Delete calibration cache
        if os.path.exists("calibration.cache"):
            os.remove("calibration.cache")

        input_tensors = [
            tensor.cuda()
            if tensor.dtype != torch.int64
            else tensor.to(torch.int32).cuda()
            for tensor in input_data.get_list(1)[0]
        ]

        model = PytorchTensorRTInferenceLearner(
            torch_model=trt_model,
            network_parameters=model_params,
            input_tfms=input_tfms,
            input_data=input_tensors if input_data is not None else None,
            dtype=dtype,
        )

        if quantization_type is not None:
            inputs, ys = input_data.get_list(100, shuffle=True, with_ys=True)
            inputs = [
                [
                    tensor.cuda()
                    if tensor.dtype != torch.int64
                    else tensor.to(torch.int32).cuda()
                    for tensor in tensors
                ]
                for tensors in inputs
            ]
            output_data = model_outputs
            is_valid = check_precision(
                model,
                inputs,
                output_data,
                metric_drop_ths,
                metric_func=metric,
                ys=ys,
            )
            if not is_valid:
                return None
        return model
