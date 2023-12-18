# whisper-cpp-pybind: python bindings for whisper.cpp

![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/sphantix/whisper-cpp-pybind/build_and_publish.yml)
![GitHub](https://img.shields.io/github/license/sphantix/whisper-cpp-pybind)
![PyPI](https://img.shields.io/pypi/v/whisper-cpp-pybind)

whisper-cpp-pybind provides an interface for calling [whisper.cpp](https://github.com/ggerganov/whisper.cpp) in Python. And whisper.cpp provides accelerated inference for [whisper](https://github.com/openai/whisper) models. This project provides both high-level and low-level API. The high-level API almost implement all the features of the [main example](https://github.com/ggerganov/whisper.cpp/tree/master/examples/main) of whisper.cpp

## Installation

### Install form PyPI
```bash
pip intall whisper-cpp-pybind
```

### Install Locally

```bash
git clone --recurse-submodules https://github.com/sphantix/whisper-cpp-pybind.git
cd whisper-cpp-python

# Install with pip
pip install .
```

### Install with Hardware Acceleration
whisper.cpp supports multiple hardware for faster processing, so you can also install with the following command:

#### OpenBLAS
Make sure you have installed `openblas`: https://www.openblas.net/

```bash
# Install with with haradware acceleration (OpenBLAS)
pip install --config-settings="--build-option=--accelerate=openblas" .
```
#### cuBLAS
Make sure you have installed `cuda` for Nvidia cards: https://developer.nvidia.com/cuda-downloads

```bash
# Install with with haradware acceleration (cuBLAS)
pip install --config-settings="--build-option=--accelerate=cublas" .
```

#### CLBlast
For cards and integrated GPUs that support OpenCL, whisper.cpp can be largely offloaded to the GPU through CLBlast. This is especially useful for users with AMD APUs or low end devices for up to ~2x speedup.

Make sure you have installed `CLBlast` for your OS: https://github.com/CNugteren/CLBlast

```bash
# Install with with haradware acceleration (CLBlast)
pip install --config-settings="--build-option=--accelerate=clblast" .
```

#### CoreML
On Apple Silicon devices, whisper.cpp can be executed on the Apple Neural Engine (ANE) via Core ML.

Make sure you have installed `CoreML` environment for your OS: https://github.com/apple/coremltools

```bash
# Install with with haradware acceleration (CoreML)
pip install --config-settings="--build-option=--accelerate=coreml" .
```

#### OpenVINO
On Intel devices which have x86 CPUs and Intel GPUs (integrated & discrete) whisper.cpp can be accelerated using OpenVINO.

Make sure you have installed `OpenVINO` environment for your OS: https://github.com/openvinotoolkit/openvino

```bash
# Install with with haradware acceleration (OpenVINO)
pip install --config-settings="--build-option=--accelerate=openvino" .
```

## Usage

### High-level API

The high-level API provides two main interface through the `Wisper` class.

Below is a simple example demonstrating how to use the high-level API to transcribe a wav file:

```python
from whisper_cpp import Whisper

whisper = Whisper("/../models/ggml-large.bin")

whisper.transcribe("samples.wav", diarize=True)

whisper.output(output_csv=True, output_jsn=True, output_lrc=True, output_srt=True, output_txt=True, output_vtt=True, log_score=True)
```

### Low-level API

All functions provided by `whisper.h` are translated to python interfaces.

Below is an example to use low-level api to transcribe.

```python
from ctypes import (
    c_float,
)
from whisper_cpp.whisper_cpp import (
    whisper_init_from_file,
    whisper_full_default_params,
    whisper_full_parallel,
    whisper_full_n_segments,
    whisper_full_get_segment_text,
    WHISPER_SAMPLING_GREEDY,
)
from whisper_cpp.utils import read_wav

pcmf32 = []
pcmf32s = []

if not read_wav("samples/samples.wav", pcmf32, pcmf32s, False):
    raise RuntimeError("Failed to read WAV file!")

whisper_ctx = whisper_init_from_file("../models/ggml-large.bin".encode("utf-8"))

wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)

pcmf32_array = (c_float * len(pcmf32))(*(i for i in pcmf32))

whisper_full_parallel(
    ctx=whisper_ctx,
    params=wparams,
    samples=pcmf32_array,
    n_samples=len(pcmf32),
    n_processors=1
)

result = ""
n_segments = whisper_full_n_segments(whisper_ctx)
for i in range(n_segments):
    text = whisper_full_get_segment_text(whisper_ctx, i)
    result = result + text.decode('utf-8')

print(result)
```
## Development

This package is under active development and any contributions will be welcomed.

## License

whisper-cpp-pybind is released under the MIT License.