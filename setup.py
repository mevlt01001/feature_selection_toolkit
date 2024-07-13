import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="feature_selection_toolkit",
    version="0.2.2",
    author="Mevlüt Başaran",
    author_email="mevlutbasaran01@gmail.com",
    description="A comprehensive toolkit for feature selection in machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mevlt01001/feature_selection_toolkit",
    project_urls={
        "Bug Tracker": "https://github.com/mevlt01001/feature_selection_toolkit/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "tqdm",
        "joblib",
        "matplotlib",
        "statsmodels",
        "xgboost"
    ],
)

