from setuptools import setup, find_packages

setup(
    name="real_estate_eda",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn"
    ],
)
