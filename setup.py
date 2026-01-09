from setuptools import setup, find_packages

setup(
    name="physics-agi",
    version="1.0.0",
    description="Model-Based Reinforcement Learning with World Models for Physical Understanding",
    author="Project Physics-AGI Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "gymnasium>=0.28.0",
        "mujoco>=2.3.0",
        "pyyaml>=6.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
    ],
)
