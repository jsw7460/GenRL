from setuptools import setup, find_packages

setup(
    name="genrl",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.23.5",
        "gymnasium>=0.29.0"
    ],
    author="Shin",
    author_email="jsw7460@gmail.com",
    description="Imitation / Offline RL framework for generalization of RL",
    license="MIT"
)