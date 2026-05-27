"""setup.py — CAREF EMNLP 2026 reproduction package."""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="caref",
    version="1.0.0",
    author="Teerapong Panboonyuen",
    author_email="teerapong.pa@chula.ac.th",
    description=(
        "CAREF: Calibration-Aware Regularization for Explanation Faithfulness "
        "Without Rationale Supervision — EMNLP 2026"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://kaopanboonyuen.github.io/CAREF",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "caref-train=scripts.train:main",
            "caref-grid=scripts.grid_search:main",
        ]
    },
)
