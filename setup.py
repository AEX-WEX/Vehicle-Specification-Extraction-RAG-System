"""
Setup configuration for vehicle-spec-rag package.
"""

from setuptools import setup, find_packages
from pathlib import Path
import re

# Read README
this_directory = Path(__file__).parent
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")
else:
    long_description = "Vehicle specification extraction from service manuals using RAG"

# Read version from src/__init__.py
version = "1.0.0"
init_path = this_directory / "src" / "__init__.py"
if init_path.exists():
    init_content = init_path.read_text(encoding="utf-8")
    version_match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', init_content, re.MULTILINE)
    if version_match:
        version = version_match.group(1)

# Core dependencies
install_requires = [
    # PDF Processing
    "pdfminer.six>=20221105",
    "PyMuPDF>=1.23.8",
    
    # ML and Embeddings
    "sentence-transformers>=2.2.2",
    "faiss-cpu>=1.7.4",
    "torch>=2.0.0",
    "transformers>=4.30.0",
    
    # LLM Integration
    "openai>=1.12.0",
    "anthropic>=0.18.1",
    
    # API Framework
    "fastapi>=0.109.2",
    "uvicorn[standard]>=0.27.1",
    "pydantic>=2.6.1",
    "pydantic-settings>=2.1.0",
    "python-multipart>=0.0.9",
    
    # Utilities
    "python-dotenv>=1.0.1",
    "pyyaml>=6.0.1",
    "numpy>=1.24.3,<2.0.0",
    "pandas>=2.0.3",
    "tqdm>=4.66.1",
    
    # Testing and Evaluation
    "scikit-learn>=1.3.2",
]

# Development dependencies
dev_requires = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.4",
    "pytest-cov>=4.1.0",
    "black>=24.1.1",
    "flake8>=7.0.0",
    "mypy>=1.8.0",
    "isort>=5.13.2",
]

# Documentation dependencies
docs_requires = [
    "sphinx>=7.2.6",
    "sphinx-rtd-theme>=2.0.0",
    "sphinx-autodoc-typehints>=1.25.2",
]

setup(
    name="vehicle-spec-rag",
    version=version,
    author="Vehicle Spec RAG Team",
    author_email="contact@example.com",
    description="RAG-based vehicle specification extraction from service manuals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vehicle-spec-rag",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/vehicle-spec-rag/issues",
        "Source": "https://github.com/yourusername/vehicle-spec-rag",
        "Documentation": "https://github.com/yourusername/vehicle-spec-rag#readme",
    },
    
    # Package configuration
    packages=find_packages(include=["src", "src.*", "api", "api.*"]),
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "docs": docs_requires,
        "all": dev_requires + docs_requires,
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "vehicle-spec-rag=main:main",
            "vsr=main:main",  # Short alias
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    
    # Keywords for PyPI
    keywords=[
        "rag",
        "retrieval-augmented-generation",
        "llm",
        "vehicle-specifications",
        "pdf-extraction",
        "faiss",
        "sentence-transformers",
        "automotive",
        "service-manual",
        "specification-extraction",
    ],
    
    # Additional metadata
    zip_safe=False,
    platforms="any",
)
