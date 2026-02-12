from setuptools import setup, find_packages
import os

# Read requirements
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

# Read README if it exists
long_description = "Real-Time GPU-Accelerated Image Filter Application"
if os.path.exists('README.md'):
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name="gpu-image-filters",
    version="1.0.0",
    description="Real-Time GPU-Accelerated Image Filter Application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'gpu-filters=main:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    keywords="gpu cuda image-processing opencv computer-vision real-time filters",
)
