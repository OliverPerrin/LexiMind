from setuptools import setup, find_packages

setup(
    name="leximind",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "scikit-learn>=1.4.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "web": [
            "streamlit>=1.25.0",
            "plotly>=5.18.0",
        ],
        "api": [
            "fastapi>=0.110.0",
        ],
        "all": [
            "streamlit>=1.25.0",
            "plotly>=5.18.0",
            "fastapi>=0.110.0",
        ],
    },
)