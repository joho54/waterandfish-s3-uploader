#!/usr/bin/env python3
"""
S3 Uploader Package Setup
미디어파이프 시퀀스 추출 및 S3 스트리밍 업로드 패키지
"""

from setuptools import setup, find_packages
import os

# README 파일 읽기
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "미디어파이프 시퀀스 추출 및 S3 스트리밍 업로드 패키지"

# requirements.txt 읽기
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="s3-uploader",
    version="1.0.0",
    author="SaturdayDinner",
    author_email="your-email@example.com",
    description="미디어파이프 시퀀스 추출 및 S3 스트리밍 업로드 파이프라인",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/s3-uploader",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "s3-uploader=s3_uploader.mediapipe_s3_streaming_pipeline:main",
            "s3-uploader-config=s3_uploader.pipeline_config:create_config_file",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md", "*.txt"],
    },
    keywords=[
        "mediapipe",
        "pose-detection",
        "sign-language",
        "s3",
        "aws",
        "video-processing",
        "machine-learning",
        "computer-vision",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/s3-uploader/issues",
        "Source": "https://github.com/your-username/s3-uploader",
        "Documentation": "https://github.com/your-username/s3-uploader#readme",
    },
) 