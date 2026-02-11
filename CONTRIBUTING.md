# Contributing to Vehicle Specification Extraction RAG System

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for our community standards.

## How to Contribute

### Reporting Bugs

Before submitting a bug report:

1. **Check existing issues** - Search closed and open issues
2. **Provide detailed information**:
   - OS and Python version
   - Equipment specifications (RAM, disk space)
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages and logs

**Creating a Bug Report**:

```
Title: [BUG] Brief description

Environment:
- Python: 3.x.x
- OS: [Windows/Linux/macOS]
- RAM: XGB

Description:
Clear description of the bug

Steps to Reproduce:
1. Step one
2. Step two
3. ...

Expected Behavior:
What should happen

Actual Behavior:
What actually happens

Logs/Errors:
```

### Suggesting Enhancements

Before suggesting an enhancement:

1. **Check if already proposed** - Search existing issues
2. **Provide clear use case** - Why is this needed?
3. **Include examples** - Show desired behavior

**Enhancement Proposal**:

```
Title: [FEATURE] Brief description

Use Case:
Why is this needed?

Proposed Solution:
How should it work?

Alternatives:
Other approaches considered

Additional Context:
Examples, mockups, references
```

### Pull Requests

#### Setup Development Environment

```bash
# Fork and clone
git clone https://github.com/yourusername/vehicle-spec-rag.git
cd vehicle-spec-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

#### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or for bugs:
   git checkout -b fix/issue-description
   ```

2. **Make your changes**
   - Write clear, self-documenting code
   - Add type hints where applicable
   - Include docstrings for public functions
   - Follow PEP 8 style guide

3. **Write tests**
   ```bash
   # Add tests in tests/
   pytest tests/
   pytest tests/ -v --cov=src
   ```

4. **Run quality checks**
   ```bash
   # Format code
   black src/ api/

   # Lint
   flake8 src/ api/

   # Type checking
   mypy src/ api/

   # All checks
   make lint
   ```

5. **Update documentation**
   - Update README if adding features
   - Update CHANGELOG.md
   - Add docstrings for new functions
   - Update API docs if applicable

6. **Commit with clear messages**
   ```bash
   git commit -m "feat: add specification validation

   - Implement validator for extraction output
   - Add comprehensive test coverage
   - Update API documentation

   Fixes #123"
   ```

7. **Push and create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

#### Pull Request Guidelines

**Title Format**:
- `feat: add new feature`
- `fix: resolve bug description`
- `docs: update documentation`
- `refactor: improve code structure`
- `perf: optimize performance`
- `test: add test coverage`
- `chore: maintenance tasks`

**Description Template**:
```markdown
## Description
Clear description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Related Issues
Fixes #123

## Testing
How was this tested?

## Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Commits follow convention
```

#### Review Process

1. **Automated Checks**
   - CI/CD pipeline runs tests
   - Code coverage must not decrease
   - Linting and type checking pass

2. **Maintainer Review**
   - Code quality assessment
   - Design review
   - Testing adequacy
   - Documentation completeness

3. **Merge**
   - Typically squashed and merged
   - Maintainers handle merge conflicts

## Development Workflow

### File Structure

```
vehicle-spec-rag/
├── src/                    # Core modules
│   ├── pipeline.py        # Main RAG pipeline
│   ├── pdf_loader.py      # PDF processing
│   ├── embeddings.py      # Embedding models
│   └── ...
├── api/                   # FastAPI application
├── tests/                 # Test suite
├── static/               # Web UI
└── docs/                 # Documentation
```

### Code Style

- **Formatting**: Black (line length: 100)
- **Linting**: Flake8
- **Type hints**: Required for public APIs
- **Docstrings**: Google style

**Example**:
```python
def extract_specifications(
    query_text: str,
    top_k: int = 5,
    return_contexts: bool = False
) -> Dict[str, Any]:
    """Extract specifications from indexed documents.

    Args:
        query_text: Natural language query for specifications
        top_k: Number of results to retrieve
        return_contexts: Whether to return retrieved context chunks

    Returns:
        Dictionary containing specifications and metadata

    Raises:
        ValueError: If query is empty
        RuntimeError: If index is not loaded
    """
```

### Testing

**Test Structure**:
```python
import pytest
from src.pipeline import VehicleSpecRAGPipeline

class TestPipeline:
    @pytest.fixture
    def pipeline(self):
        return VehicleSpecRAGPipeline(config_path="config.yaml")

    def test_initialization(self, pipeline):
        assert pipeline is not None
        assert pipeline.initialized

    @pytest.mark.skip(reason="requires PDF file")
    def test_query(self, pipeline):
        result = pipeline.query("brake torque")
        assert "specifications" in result
```

**Coverage Requirements**:
- Minimum 80% coverage for new code
- 100% coverage for critical paths
- Run: `pytest --cov=src tests/`

### Documentation

**Types of Documentation**:

1. **Code Comments** - Explain *why*, not *what*
2. **Docstrings** - Public function/class documentation
3. **README** - Project overview
4. **CONTRIBUTING** - This file
5. **API_DOCUMENTATION** - API reference
6. **ARCHITECTURE** - System design

**Documentation Style**:
- Clear and concise
- Use examples
- Link to related docs
- Keep up to date

## Release Process

### Version Numbering

Follows [Semantic Versioning](https://semver.org/):
- MAJOR.MINOR.PATCH (e.g., 1.2.3)
- Breaking changes → MAJOR
- New features (backward compatible) → MINOR
- Bug fixes → PATCH

### Release Checklist

1. Update version in `setup.py` and `__init__.py`
2. Update `CHANGELOG.md` with changes
3. Ensure all tests pass
4. Create Git tag: `git tag -a v1.2.3 -m "Release v1.2.3"`
5. Push tag: `git push origin v1.2.3`
6. Create GitHub Release with changelog

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Issues**: Check existing issues or open a new one
- **Documentation**: See [docs/](docs/) folder
- **Slack**: [Community Slack Channel] (if available)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors are recognized in:
- Git commit history
- GitHub contributors page
- CHANGELOG.md for significant contributions
- Authors section in documentation

Thank you for contributing!
