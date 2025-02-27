from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="titans",
    version="0.1.0",
    author=["Priyanshu Singh","Prikshit Singh"],
    author_email=["singhpriyanshu783@gmail.com","prikshitsingh79@gmail.com"],
    description="Titans: Learning to Memorize at Test Time",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wolverinex24/titans",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GPL-3.0",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=1.13.0",
        "transformers>=4.25.0",
        "numpy>=1.20.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mkdocs>=1.3.0",
            "mkdocs-material>=8.2.0",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
        ],
    },
)