from pathlib import Path

from setuptools import find_namespace_packages, setup


ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="sysidmjx",
    version="0.1.0",
    description="Differentiable single-motor system identification with MuJoCo MJX",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Vyacheslav Kovalev",
    author_email="ssaavvaa222@gmail.com",
    url="https://github.com/Slavoch/mjx_sysid",
    project_urls={
        "Documentation": "https://github.com/Slavoch/mjx_sysid#readme",
        "Source": "https://github.com/Slavoch/mjx_sysid",
        "Issues": "https://github.com/Slavoch/mjx_sysid/issues",
    },
    license="Apache-2.0",
    packages=find_namespace_packages(
        include=["sysidmjx", "sysidmjx.*", "config", "config.*"],
    ),
    install_requires=[
        "flax",
        "jax",
        "matplotlib",
        "mujoco",
        "mujoco-mjx",
        "numpy",
        "optax",
        "pandas",
        "scipy",
        "tensorboard",
        "tensorboardX",
    ],
    python_requires=">=3.10",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    keywords=[
        "jax",
        "mujoco",
        "mjx",
        "robotics",
        "system-identification",
    ],
)
