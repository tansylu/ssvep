from setuptools import setup, find_packages

setup(
    name="ssvep",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "pandas",
        "pillow",
        "imageio",
        "opencv-python",
        "seaborn",
    ],
    author="SSVEP Team",
    author_email="example@example.com",
    description="A tool for analyzing neural network activations in response to flickering images",
    keywords="neural networks, fft, signal processing",
    python_requires=">=3.6",
)
