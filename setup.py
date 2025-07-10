from setuptools import setup, find_packages

setup(
    name="wandbproxy",
    version="0.1.0",
    description="Sync wandb ops to others platforms.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        # 你的依赖列表，例如：
        # 'numpy>=1.18.0',
        # 'pandas>=1.0.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)