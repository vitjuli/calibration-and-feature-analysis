# Contributing to Electronic Nose ML Research

Thank you for your interest in contributing to this research project! This document provides guidelines for contributions.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check existing issues to avoid duplicates
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details (Python version, OS, etc.)

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add docstrings to new functions/classes
   - Include type hints where applicable
   - Add tests if adding new functionality

4. **Test your changes**
   ```bash
   python -m pytest tests/
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "Add: brief description of changes"
   ```

6. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

### Python Style Guide
- Follow PEP 8
- Use meaningful variable names
- Maximum line length: 88 characters (Black formatter)
- Use type hints for function signatures

### Documentation Style
- Docstrings: Google style
- Include:
  - Brief description
  - Args section
  - Returns section
  - Example usage (when helpful)

Example:
```python
def compute_importance(
    model: nn.Module,
    data_loader: DataLoader
) -> np.ndarray:
    """
    Compute feature importance scores.

    Args:
        model: Trained neural network model
        data_loader: DataLoader with input data

    Returns:
        Array of importance scores for each feature

    Example:
        >>> scores = compute_importance(model, train_loader)
        >>> print(f"Top feature: {scores.argmax()}")
    """
    pass
```

## Research Contributions

### Adding New Methods

If you implement a new interpretability or calibration method:

1. Add implementation in appropriate `src/` folder
2. Create a notebook demonstrating the method
3. Update README with method description
4. Include references to original papers
5. Add comparison with existing methods

### Experiments

When adding new experiments:

1. Document experimental setup
2. Include random seeds for reproducibility
3. Report mean ± std over multiple runs
4. Compare with baseline methods
5. Visualize results

## Testing

### Required Tests
- Unit tests for new functions
- Integration tests for workflows
- Reproducibility tests (same seed → same results)

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_deep_taylor.py

# Run with coverage
python -m pytest --cov=src tests/
```

## Documentation

### Updating Documentation
- Update relevant README files
- Add examples for new features
- Document breaking changes
- Update requirements.txt if adding dependencies

### Notebooks
- Clear all outputs before committing
- Add markdown cells explaining steps
- Include visualization of results
- Ensure notebooks run from top to bottom

## Data Contributions

### Providing New Data
If you have similar gas sensor data:

1. Document collection protocol
2. Include metadata (sensors, gases, conditions)
3. Specify data format
4. Provide preprocessing scripts
5. Note any data use restrictions

## Questions?

Contact: iv294@cam.ac.uk

## Code of Conduct

### Our Standards
- Be respectful and inclusive
- Accept constructive criticism
- Focus on what's best for the research community
- Show empathy towards others

### Unacceptable Behavior
- Harassment or discriminatory language
- Personal attacks
- Publishing others' private information
- Other unprofessional conduct

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
