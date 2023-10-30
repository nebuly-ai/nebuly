# Advanced options

If you’re new to the library, you may want to start with the **Getting started** section.

The user guide here shows more advanced workflows and how to use the library in different ways. We are going to show some examples of more advanced usages of `Speedster`, that we hope will give you a deeper insight of how `Speedster` works. 

In particular, we will overview:

- [`optimize_model`](#optimizemodel-api) API
- [Acceleration suggestions](#acceleration-suggestions)
- [Selecting which device](#selecting-which-device-to-use--cpu-gpu-and-other-accelerators) to use: CPU, GPU and other accelerators
- [Optimization Time: constrained vs unconstrained](#optimization-time--constrained-vs-unconstrained)
- [Selecting specific compilers/compressors](#select-specific-compilerscompressors)
- [Using dynamic shape](#using-dynamic-shape)
- [Enable TensorrtExecutionProvider for ONNXRuntime on GPU](#enable-tensorrtexecutionprovider-for-onnxruntime-on-gpu)
- [Custom models](#custom-models)
- [Store the performances of all the optimization techniques](#store-the-performances-of-all-the-optimization-techniques)
- [Set number of threads](#set-number-of-threads)

## `optimize_model` API

The `optimize_model` function allows to optimize a model from one of the supported frameworks (PyTorch, HuggingFace, TensorFlow, ONNX), and returns an optimized model that can be used with the same interface as the original model.

```python
def optimize_model(
        model: Any,
        input_data: Union[Iterable, Sequence],
        metric_drop_ths: Optional[float] = None,
        metric: Union[str, (...) -> Any, None] = None,
        optimization_time: str = "constrained",
        dynamic_info: Optional[dict] = None,
        config_file: Optional[str] = None,
        ignore_compilers: Optional[List[str]] = None,
        ignore_compressors: Optional[List[str]] = None,
        store_latencies: bool = False,
        device: str = None,
        **kwargs: Any
) -> Any
```

**Arguments**

`model`: Any

The input model can belong to one of the following frameworks: PyTorch, TensorFlow, ONNX, HuggingFace. In the ONNX case, `model` is a string with the path to the saved onnx model. In the other cases, it is a torch.nn.Module or a tf.Module.

`input_data`: Iterable or Sequence

Input data needed to test the optimization performances (latency, throughput, accuracy loss, etc). It can consist of one or more data samples. Note that if `optimization_time` is set to "unconstrained," it would be preferable to provide at least 100 data samples to also activate `Speedster` techniques that require more data (pruning, etc.). See the Getting started section to learn more about the `input_data` depending on your input framework:

- [Getting started with PyTorch optimization](getting_started/pytorch_getting_started.md#1-input-model-and-data)
- [Getting started with 🤗 HuggingFace optimization](getting_started/hf_getting_started.md#1-input-model-and-data)
- [Getting started with Stable Diffusion optimization](getting_started/diffusers_getting_started.md#1-input-model-and-data)
- [Getting started with TensorFlow/Keras optimization](getting_started/tf_getting_started.md#1-input-model-and-data)
- [Getting started with ONNX optimization](getting_started/onnx_getting_started.md#1-input-model-and-data)

`metric_drop_ths`: float, optional

Maximum drop in your preferred metric (see "metric" section below). All the optimized models having a larger error with respect to the `metric_drop_ths` will be discarded. 

Default: 0.

`metric`: Callable, optional

Metric to be used for estimating the error that may arise from using optimization techniques and for evaluating if the error exceeds the `metric_drop_ths`.  `metric` accepts as input a string, a user-defined metric, or None. Metric accepts a string containing the name of the metric; it currently supports:

- "numeric_precision"
- "accuracy". 
- user-defined metric: function that takes as input the output of the original model and the one of the optimized model, and, if available, the original label. The function calculates and returns the reduction in the metric due to the optimization. 

Default: "numeric_precision". 

`optimization_time`: OptimizationTime, optional

The optimization time mode. It can be "constrained" or "unconstrained". In "constrained" mode, Speedster takes advantage only of compilers and precision reduction techniques, such as quantization. "unconstrained" optimization_time allows it to exploit more time-consuming techniques, such as pruning and distillation. Note that most techniques activated in "unconstrained" mode require fine-tuning, and therefore it is recommended to provide at least 100 samples as input_data. 

Default: "constrained".

`dynamic_info`: Dict, optional

Dictionary containing dynamic axis information. It should contain as keys both "input" and "output" and as values two lists of dictionaries, where each dictionary represents dynamic axis information for an input/output tensor. The inner dictionary should have an integer as a key, i.e. the dynamic axis (also considering the batch size) and a string as a value giving it a tag, e.g., "batch_size.". 

Default: None.

`config_file`: str, optional

Configuration file containing the parameters needed to define the CompressionStep in the pipeline. 

Default: None.

`ignore_compilers`: List[str], optional

List of DL compilers ignored during optimization execution. The compiler name should be one among tvm, tensor RT, openvino, onnxruntime, deepsparse, tflite, bladedisc, torchscript, intel_neural_compressor . 

Default: None.

`ignore_compressors`: List[str], optional

List of DL compressors ignored during the compression stage. The compressor name should be one among sparseml and intel_pruning. 

Default: None.

`store_latencies`: bool, optional

Parameter that allows to store the latency for each compiler used by Speedster in a json file. The JSON is created in the working directory. 

Default: False.

`device`: str, optional

Device used for inference, it can be cpu or gpu/cuda (both gpu and cuda options are supported). A specific gpu can be selected using notation gpu:1 or cuda:1. gpu will be used if available, otherwise cpu. 

Default: None.

**Returns: Inference Learner**

Optimized version with the same interface of the input model. For example, optimizing a PyTorch model will return an InferenceLearner object that can be called exactly like a PyTorch model (either with model.forward(input) or model(input)). The optimized model will therefore take as input a torch.Tensors and return a torch.Tensors.

## Acceleration suggestions

If the speedup you obtained with the first optimization with `Speedster` is not enough, we suggest the following actions:

- Include more backends for optimization, i.e. set `--backend all`
- Increase the `metric_drop_ths` by 5%, if possible: see [Optimize_model API](#optimize_model-api)
- Verify that your device is supported by your version of speedster: see [Supported hardware](hardware.md)
- Try to accelerate your model on a different hardware or consider using the CloudSurfer module to automatically understand which is the best hardware for your model: see [CloudSurfer](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/cloud_surfer) module.

## Selecting which device to use: CPU, GPU and other accelerators.

Speedster currently supports the following devices: `CPUs`, `GPUs`, `TPUs` and `AWS Inferentia chips`.

The parameter `device` allows to select which device we want to use for inference. By default, `Speedster` will use the accelerator if available on the machine, otherwise it will use cpu. If we are running on a machine with an available accelerator and we want to optimize the model for cpu inference, we can use:

```python
from speedster import optimize_model

optimized_model = optimize_model(
  model, input_data=input_data, device="cpu"
)
```

If we are working on a multi-gpu machine and we want to use a specific gpu, we can use:

```python
from speedster import optimize_model

optimized_model = optimize_model(
  model, input_data=input_data, device="cuda:1"  # also device="gpu:1" is supported
)
```

The same applies also for TPUs and AWS Inferentia chips: 

```python
from speedster import optimize_model

optimized_model = optimize_model(
  model, input_data=input_data, device="tpu:1"  # use tpu #1
)

optimized_model = optimize_model(
  model, input_data=input_data, device="neuron:1"  # use Inferentia chip #1
)
```

## Optimization Time: constrained vs unconstrained

One of the first options that can be customized in `Speedster` is the `optimization_time` parameter. In order to optimize the model, `Speedster` will try a list of compilers which allow to keep the same accuracy of the original model. In addition to compilers, it can also use other techniques such as pruning, quantization, and other compression techniques which can lead to a little drop in accuracy and may require some time to complete. 

We defined two scenarios:

- **constrained**: only compilers and precision reduction techniques are used, so the compression step (the most time consuming one) is skipped. Moreover, in some cases the same compiler could be available for more than one pipeline, for example tensor RT is available both with PyTorch and ONNX backends. In the constrained scenario, each compiler will be used only once, so if for example we optimize a PyTorch model and tensor RT in the PyTorch pipeline manages to optimize the model, it won't be used again in the ONNX pipeline.

- **unconstrained**: in this scenario, `Speedster` will use all the compilers available, even if they appear in more than one backend. It also allows the usage of more time consuming techniques such as pruning and distillation. Note that for using many of the sophisticated techniques in the 'unconstrained' optimization, a small fine-tuning of the model will be needed. Thus, we highly recommend to provide as input_data at least 100 samples when selecting 'unconstrained' optimization.


##  Select specific compilers/compressors

The `optimize_model` functions accepts also the parameters `ignore_compilers` and `ignore_compressors`, which allow to skip specific compilers or compressors. 
The full list of available options is the following:
- _ignore_compilers_: `deepsparse`, `tensor_rt`, `torch_tensor_rt`, `onnx_tensor_rt`, `torchscript`, `onnxruntime`, `tflite`, `tvm`, `onnx_tvm`, `torch_tvm`, `bladedisc`, `openvino`, `intel_neural_compressor`, `torch_xla`, `torch_neuron`.
- _ignore_compressors_: `sparseml`, `intel_pruning`.

Some compilers, such as tensor RT, are available for both PyTorch and ONNX backends. For this reason in the list of compilers we have `tensor_rt` which skips both the PyTorch and ONNX pipelines, and `torch_tensor_rt` and `onnx_tensor_rt` which skip only the PyTorch and ONNX pipelines respectively.

If we want to skip the `tvm` and `bladedisc` optimizers, we could write:

```python
from speedster import optimize_model

optimized_model = optimize_model(
    model, 
    input_data=input_data, 
    ignore_compilers=["tvm", "bladedisc"]
)
```

## Using dynamic shape

By default, a model optimized with `Speedster` will have a static shape. This means that it can be used in inference only with the same shape of the inputs provided to the `optimize_model` function during the optimization. The dynamic shape however is fully supported, and can be enabled with the `dynamic_info` parameter (see the [optimize_model API](#optimize_model-api) arguments to see how this parameter is defined.)

For each dynamic axis in the inputs, we need to provide the following information:
- the axis number (starting from 0, considering the batch size as the first axis)
- a tag that will be used to identify the axis
- the minimum, optimal and maximum sizes of the axis (some compilers will work also for shapes that are not in the range [min, max], but the performance may be worse)

Let's see an example of a model that takes two inputs, where the batch size must be dynamic, as well as the size on the third and fourth dimensions.

```python
import torch
import torchvision.models as models
from speedster import optimize_model

# Load a resnet as example
model = models.resnet50()

# Provide an input data for the model
input_data = [((torch.randn(1, 3, 256, 256),), torch.tensor([0])) for _ in range(100)]

# Set dynamic info
dynamic_info = {
    "inputs": [
        {
            0: {
                "name": "batch",
                "min_val": 1,
                "opt_val": 1,
                "max_val": 8,
            }, 
            2: {
                "name": "dim_image",
                "min_val": 128,
                "opt_val": 256,
                "max_val": 512,
            }, 
            3: {
                "name": "dim_image",
                "min_val": 128,
                "opt_val": 256,
                "max_val": 512,
            }, 
        }
    ],
    "outputs": [
        {0: "batch", 1: "out_dim"}
    ]
}

# Run Speedster optimization in one line of code
optimized_model = optimize_model(
    model, 
    input_data=input_data, 
    optimization_time="constrained", 
    dynamic_info=dynamic_info
)
```

## Enable TensorrtExecutionProvider for ONNXRuntime on GPU

By default, `Speedster` will use the `CUDAExecutionProvider` for ONNXRuntime on GPU. If you want to use the `TensorrtExecutionProvider` instead, you must add the TensorRT installation path to the env variable LD_LIBRARY_PATH.
If you installed TensorRT through the nebullvm auto_installer, you can do it by running the following command in the terminal:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/<PATH_TO_PYTHON_FOLDER>/site-packages/tensorrt"
```

## Custom models

`Speedster` is designed to optimize models that take as inputs and return in output only tensors or np.ndarrays (and dictionaries/strings for huggingface). Some models may require instead a custom input, for example a dictionary where the keys are the names of the inputs and the values are the input tensors, or may return a dictionary as output. We can optimize such models with `Speedster` by defining a model wrapper.

Let's take the example of the detectron2 model which takes as input a tuple of tensors but returns a dictionary as output:

```python
 class BaseModelWrapper(torch.nn.Module):
    def __init__(self, core_model, output_dict):
        super().__init__()
        self.core_model = core_model
        self.output_names = [key for key in output_dict.keys()]
    
    def forward(self, *args, **kwargs):
        res = self.core_model(*args, **kwargs)
        return tuple(res[key] for key in self.output_names)


class OptimizedWrapper(torch.nn.Module):
    def __init__(self, optimized_model, output_keys):
        super().__init__()
        self.optimized_model = optimized_model
        self.output_keys = output_keys
    
    def forward(self, *args):
        res = self.optimized_model(*args)
        return {key: value for key, value in zip(self.output_keys, res)}

input_data = [((torch.randn(1, 3, 256, 256)), torch.tensor([0]))]

# Compute the original output of the model (in dict format) 
res = model_backbone(torch.randn(1, 3, 256, 256))

# Pass the model and the output sample to the wrapper
backbone_wrapper = BaseModelWrapper(model_backbone, res)

# Optimize the model wrapper
optimized_model = optimize_model(backbone_wrapper, input_data=input_data)

# Wrap the optimized model with a new wrapper to restore the original model output format
optimized_backbone = OptimizedWrapper(optimized_model, backbone_wrapper.output_names)

```

You can find other examples in the [notebooks](https://github.com/nebuly-ai/nebuly/tree/main/optimization/speedster/notebooks) section available on GitHub.

## Store the performances of all the optimization techniques

`Speedster` internally tries all the techniques available on the target hardware and automatically chooses the fastest one. If you need more details on the inference times of each compiler, you can set the `store_latencies` parameter to `True`. A json file will be created in the working directory, listing all the results of the applied techniques and of the original model itself.

```python
# Run Speedster optimization in one line of code
optimized_model = optimize_model(
    model, 
    input_data=input_data, 
    store_latencies=True
)
```

## Set number of threads
When running multiple replicas of the model in parallel, it would be useful for CPU-optimized algorithms to limit the number of threads to use for each model. In `Speedster`, it is possible to set the maximum number of threads a single model can use with the environment variable `NEBULLVM_THREADS_PER_MODEL`. 

For instance, you can run:

```python
export NEBULLVM_THREADS_PER_MODEL = 2
```

for using just two CPU threads per model at inference time and during optimization.