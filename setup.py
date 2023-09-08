from skbuild import setup
from pathlib import Path

root= Path(__file__).parent
long_description = (root / "README.md").read_text(encoding="utf-8")

setup(
    name="whisper_cpp_python",
    description="A Python wrapper for whisper.cpp",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.1.1",
    author="sphantix",
    author_email="sphantix@gmail.com",
    license="MIT",
    package_dir={"whisper_cpp": "whisper_cpp",},
    package_data={"whisper_cpp": ["py.typed"]},
    packages=["whisper_cpp",],
    install_requires=["numpy>=1.20.0"],
    extras_require={
    },
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
