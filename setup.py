from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="justshowit",
    url="https://github.com/Jako-K/justshowit",
    version='1.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Jako-K",
    description='Display images without all the nonsense!',
    packages=find_packages(),
    install_requires=["opencv-python", "requests", "numpy", "Pillow", "tqdm"],
)




