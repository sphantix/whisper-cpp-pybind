import os
import sys
import re
from pathlib import Path
from skbuild import setup

def get_accelerate_option() -> str:
    accelerate = None
    accelerate_option = [s for s in sys.argv if "--accelerate" in s]
    for element in accelerate_option:
        accelerate = re.split('[= ]',element)[1]
        sys.argv.remove(element)
    return accelerate

root= Path(__file__).parent
long_description = (root / "README.md").read_text(encoding="utf-8")
accelerate = get_accelerate_option()

if accelerate is not None:
    if accelerate == "openblas":
        os.environ["WHISPER_OPENBLAS"] = "1"
    elif accelerate == "clblast":
        os.environ["WHISPER_CLBLAST"] = "1"
    elif accelerate == "cublas":
        os.environ["WHISPER_CUBLAS"] = "1"
    elif accelerate == "openvino":
        os.environ["WHISPER_OPENVINO"] = "1"
    elif accelerate == "coreml":
        os.environ["WHISPER_COREML"] = "1"

setup(
    name="whisper-cpp-pybind",
    description="A Python wrapper for whisper.cpp",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.1.3",
    author="sphantix",
    author_email="sphantix@gmail.com",
    license="MIT",
    package_dir={"whisper_cpp": "whisper_cpp",},
    package_data={"whisper_cpp": ["py.typed"]},
    packages=["whisper_cpp",],
    install_requires=["numpy>=1.20.0"],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
