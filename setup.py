from setuptools import find_namespace_packages, setup

setup(
    name="ambirefer3d",
    version="1.0",
    author="Yiming Zhang",
    description="AmbiRefer3D: interactive 3D visual grounding",
    packages=find_namespace_packages(include=("m3drefclip*", "stage2_refactor*", "data_construction*")),
    install_requires=[
        "lightning==2.0.6", "wandb==0.15.8", "scipy", "hydra-core",
        "h5py", "open3d", "pandas", "transformers", "tqdm", "Pillow"
    ]
)
