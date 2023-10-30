# Installation
In this installation guide we will learn:

- [Quick installation](#quick-installation) of `Speedster` with pip **(Recommended)** 

- [Selective installation](#optional-selective-installation-of-speedster-requirements) of the requirements **(Optional)**

- [Installation](#optional-download-docker-images-with-frameworks-and-optimizers) with Docker **(Optional)** 

- [Set up Speedster on custom DL devices](#set-up-speedster-on-custom-dl-devices) to run models on Google TPUs and AWS Inferentia Chips


## Quick installation 
You can easily install `Speedster` using pip.

    pip install speedster

Then make sure to install all the available deep learning compilers:

    python -m nebullvm.installers.auto_installer --compilers all


!!! info
    If you want to optimize PyTorch or HuggingFace models, PyTorch must be pre-installed in the environment before using the auto-installer, please install it from [this](https://pytorch.org/get-started/locally/) link. Moreover, for Mac computers with M1/M2 processors, please use a conda environment, or you may run into problems when installing some of the deep learning compilers.

Great, now you are ready to accelerate your model 🚀 Please visit the following pages to get started based on the DL framework of your input model:

- [Getting started with PyTorch optimization](getting_started/pytorch_getting_started.md)
- [Getting started with 🤗 Hugging Face optimization](getting_started/hf_getting_started.md)
- [Getting started with Stable Diffusion optimization](getting_started/diffusers_getting_started.md)
- [Getting started with TensorFlow/Keras optimization](getting_started/tf_getting_started.md)
- [Getting started with ONNX optimization](getting_started/onnx_getting_started.md)


## (Optional) Selective installation of Speedster requirements

By default, the `auto_installer` installs all the DL frameworks and compilers supported by `Speedster`. However, some of these may not be relevant to your use case. In this section, we explain how you can customize the installation of these libraries, avoiding those that are not needed.

To customize the libraries installation you have two options:

- [Use the auto-installer (recommended)](#use-the-auto-installer-recommended)
- [Install the libraries manually](#manual-installation)

### Use the auto-installer (recommended)
To understand how to selectively install your preferred libraries, let's examine the auto-installer API:

```bash
python -m nebullvm.installers.auto_installer 
    --frameworks <frameworks> 
    --extra-backends <backends> 
    --compilers <compilers>
```

!!! Description

    === "--frameworks"

        `frameworks` is used to specify the deep learning framework of your input model. The supported frameworks are `torch`, `tensorflow`, `onnx`, `huggingface` and `diffusers`.

        - if you want to optimize a model with a single DL framework, the code is as follows (example below for HuggingFace):
            
            ```python
            python -m nebullvm.installers.auto_installer --frameworks huggingface
            ```
            
            Please remember that for PyTorch optimization, you should pre-install PyTorch from the official [repo](https://pytorch.org/get-started/locally/).
                
        - if you want to optimize models in multiple input frameworks, you can include them separated with a space:
            ```python
            python -m nebullvm.installers.auto_installer --frameworks tensorflow torch
            ```

        - If you want to include all the frameworks, you can use `all` as the argument:

            ```python
            python -m nebullvm.installers.auto_installer --frameworks all
            ```

        Default: `all`.
    
    === "--extra-backends"

        After entering your input model, `Speedster` converts the input model from its original framework into an intermediate framework to be used during the optimization; we call these intermediate frameworks "backends." To learn more, see the section [Model Converter](https://docs.nebuly.com/Speedster/key_concepts/) in the docs. This conversion allows `Speedster` to apply all optimization techniques without being constrained by the input framework of your model.
            
        The supported backends are `torch`, `tensorflow` and `onnx`.
            
        You can specify multiple backends by separating them with a space. 
            
        - For example, if you want to install TensorFlow and ONNX as backends of an HugginFace model, the code is as follows:
            
            ```python
            python -m nebullvm.installers.auto_installer --frameworks huggingface --extra-backends tensorflow onnx
            ```python
            
        - If you want to install all the backends supported by the selected frameworks, you can use `all` as the argument.
        - If you don't want to install extra backends, you can set `--extra-backends none`.
            
        The extra-backends that you choose must be compatible with at least one of the input frameworks you previously selected with the argument `—-frameworks`, please see the table below to see the compatibility matrix. 

        Default: `all`.    

    === "--compilers"

        `compilers` is used to specify the deep learning compilers to be installed. The supported compilers are: `deepsparse`, `tensor_rt`, `torch_tensor_rt`, `openvino` and `intel_neural_compressor`. The compilers must be compatible with at least one of the backends selected with the argument `—-extra-backends`, please see the table below to see the compatibility matrix.

        - You can specify multiple compilers by separating them with a space. For example:
            
            ```python
            --compilers deepsparse tensor_rt
            ```
            
            will install DeepSparse and TensorRT. 
            
        - If you want to install all the compilers supported by the selected frameworks/backends, you can use `all` as the argument.

        Speedster also supports `torchscript`, `tf_lite`, and `onnxruntime` as built-in; these are preinstalled with their respective backends, so there is no need to include them in the list. Speedster also supports `tvm`, which is currently not supported by the automatic installer and must be installed manually; see the next section if you wish to include it.

        Default: `all`.


Let's see an example of how to use these three arguments:

```bash
python -m nebullvm.installers.auto_installer 
    --frameworks torch 
    --extra-backends all 
    --compilers all
```

This command will setup your environment to optimize PyTorch models, and will install all PyTorch supported backends and compilers.

The following table shows the supported combinations of frameworks, backends and compilers that you can install with the auto-installer:

| Framework    | Extra Backends            | Compilers                                                               |
|--------------|---------------------------|-------------------------------------------------------------------------|
| PyTorch      | ONNX                      | DeepSparse, TensorRT, Torch TensorRT, OpenVINO, Intel Neural Compressor |
| TensorFlow   | ONNX                      | TensorRT, OpenVINO                                                      |
| ONNX         | /                         | TensorRT, OpenVINO                                                      |
| Hugging Face | PyTorch, TensorFlow, ONNX | DeepSparse, TensorRT, Torch TensorRT, OpenVINO, Intel Neural Compressor |
| Diffusers    | PyTorch, ONNX             | DeepSparse, TensorRT, Torch TensorRT, OpenVINO, Intel Neural Compressor |


!!! info
    Hugging Face models can be of two types, PyTorch-based or TensorFlow-based. For PyTorch-based models, it is necessary to include `torch` as an extra-backend. For TensorFlow-based models, you must include `tensorflow` as an extra-backend.

### Manual installation

If you want to manually install the requirements, this section collects links to the official installation guides for all frameworks and compilers supported by `Speedster`.

### Deep Learning frameworks/backends

- [PyTorch](https://pytorch.org/get-started/locally/)
- [TensorFlow](https://www.tensorflow.org/install)
- [ONNX](https://github.com/onnx/onnx#installation)
- [HuggingFace](https://huggingface.co/transformers/installation.html)
- [Diffusers](https://github.com/huggingface/diffusers#installation)

### Deep Learning compilers

- [DeepSparse](https://github.com/neuralmagic/deepsparse#installation)
- [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
- [Torch TensorRT](https://pytorch.org/TensorRT/getting_started/installation.html#installation)
- [ONNXRuntime](https://onnxruntime.ai/docs/install/#python-installs)
- [OpenVINO](https://docs.openvino.ai/latest/openvino_docs_install_guides_install_dev_tools.html#step-4-install-the-package)
- [Intel Neural Compressor](https://github.com/intel/neural-compressor#installation)
- [Apache TVM](https://tvm.apache.org/docs/install/index.html)

### Other requirements

- [tf2onnx](https://github.com/onnx/tensorflow-onnx#installation) (Install it if you want to convert TensorFlow models to ONNX)
- [polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy#installation) (Install it if you want to use TensorRT)
- [onnx-simplifier](https://github.com/daquexian/onnx-simplifier#python-version) (Install it if you want to use TensorRT)
- [onnx_graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon#installation) (Install it if you want to use TensorRT with Stable Diffusion)
- [onnxmltools](https://github.com/onnx/onnxmltools#install) (Install it if you want to convert models to ONNX)

## (Optional) Download Docker images with frameworks and optimizers

Instead of installing the frameworks and compilers needed for optimization, which can be a time-consuming task, you can simply download a Docker container with all compilers preinstalled.

To pull up the Docker image, run:

    docker pull nebulydocker/nebullvm:latest

and then run and access the Docker with:

    docker run -ti --gpus=all nebulydocker/nebullvm:latest

After optimizing the model, you may decide to deploy it to production. Note that you need to have the deep learning compiler used to optimize the model and other components inside the production Docker. For this reason, we have created several versions of the Docker nebullvm container in the [Docker Hub](https://hub.docker.com/r/nebulydocker/nebullvm), each containing only one compiler. Pull the image with the compiler that has optimized your model!

## Set up Speedster on custom DL devices

From version `0.10.0`, Speedster supports optimization of PyTorch models on `Google TPUs` and `AWS Inferentia` chips. 
For these devices, the user must ensure that the required libraries are installed on the machine. 
The following sections describe how to install the required libraries for each device.

### Google TPUs

In order to use a TPU, you must request a TPU-enabled VM from Google Cloud. You can consult the [official documentation](https://cloud.google.com/tpu/docs/run-calculation-pytorch?hl=en) 
for more information about how to create a TPU VM and how to get started with PyTorch on TPUs.

To use Speedster on Google TPUs, we will use the [`torch_xla`](https://github.com/pytorch/xla) library, which is already 
preinstalled in all the Google Cloud TPU VMs, you will find it in the base Python3 environment.

After creating the VM, you can follow these steps to set up Speedster:
- Check that the `torch_xla` library is installed in the base Python3 environment. You can do this by running `python3 -c "import torch_xla; print(torch_xla.__version__)"` in the VM console;
- Set TPU runtime configuration as explained in the [official documentation](https://cloud.google.com/tpu/docs/run-calculation-pytorch?hl=en#set_tpu_runtime_configuration);
- [Optional] Check that the TPU is working by running the [official example](https://cloud.google.com/tpu/docs/run-calculation-pytorch?hl=en#perform_a_simple_calculation);
- Install Speedster by running `pip install speedster`. It's not required to install the deep learning compilers in this case, since they are not supported on TPUs.

You are now ready to use Speedster on TPUs! Speedster will automatically detect the TPU device and will use the `torch_xla` library to optimize the model, comparing its performances with the original model running on the CPU.

### AWS Inferentia

For AWS Inferentia, you must first create an AWS EC2 instance with the `inf1` instance type.
You can find more information about `inf1` instances in the [official documentation](https://aws.amazon.com/it/ec2/instance-types/inf1/).

!!! info
    AWS has recently released the `inf2` instance type, which is a more powerful version of `inf1`. For now `inf2`
instances are only available in private preview, you can request them directly to AWS by filling this [form](https://pages.awscloud.com/EC2-Inf2-Preview.html).

To use Speedster on AWS Inferentia, we will use the [`torch-neuron`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-setup.html) library, that must be manually installed on `inf1` instances (on `inf2`instances it's already preinstalled if you use the PyTorch DLAMI provided by AWS).

You can find here the full guides to set up the EC2 instances and install the required libraries:

- `inf1`: <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuron/setup/pytorch-install.html#install-neuron-pytorch>

- `inf2`: <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/setup/pytorch-install.html#pytorch-neuronx-install>

After creating the EC2 instance and installing `torch_neuron`, you can follow these steps to set up Speedster:
- Check that the `torch_neuron` library is installed, you can do this by running `python -c "import torch_neuron; print(torch_neuron.__version__)"` in the console (if using `inf1` instances, otherwise change `torch_neuron` with `torch_neuronx`);
- Install Speedster by running `pip install speedster`. It's not required to install the deep learning compilers in this case, since they are not supported on AWS Inferentia.

You are now ready to use Speedster on AWS Inferentia! Speedster will automatically detect the AWS Inferentia device and will use the `torch_neuron` library to optimize the model, comparing its performances with the original model running on the CPU.
