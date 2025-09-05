# Minimal kokoro-onnx for Japanese TTS

## Installation

First, install the Python package from this repo:

``` bash
pip install kokoro_onnx@git+https://github.com/moonshine-ai/kokoro-onnx-ja.git
```

Then download the models (quantized Kokoro and style vector for Japanese speaker):

``` bash
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx -P model/ 
wget https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/jf_alpha.bin -P model/
```

Refer to `main.py` in this repo for a usage example.
