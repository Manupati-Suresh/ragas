# Contributing to RAG Quote System

Thank you for your interest in contributing to the RAG Quote System! This document provides guidelines and information for contributors.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]
   ```
5. **Run tests** to ensure everything works:
   ```bash
   python test_rag_system.py
   ```

## 📝 Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and single-purpose

### Testing

- Write tests for new functionality
- Ensure all tests pass before submitting
- Add integration tests for complex features
- Test with different Python versions (3.8+)

### Documentation

- Update README.md for new features
- Add inline comments for complex logic
- Update docstrings when changing function signatures
- Include usage examples

## 🔧 Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit them:
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

3. **Run tests** to ensure nothing is broken:
   ```bash
   python test_rag_system.py
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** on GitHub

## 🐛 Reporting Issues

When reporting issues, please include:

- **Description**: Clear description of the problem
- **Steps to reproduce**: Detailed steps to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: Python version, OS, dependencies
- **Error messages**: Full error traceback if applicable

## 💡 Feature Requests

For feature requests, please:

- Describe the feature in detail
- Explain the use case and benefits
- Provide examples if possible
- Consider implementation complexity

## 🔍 Code Review Process

1. **Automated checks** must pass (tests, linting)
2. **Code review** by maintainers
3. **Address feedback** and make requested changes
4. **Merge** when approved

## 📋 Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] Tests are written and passing
- [ ] Documentation is updated
- [ ] No sensitive data is included
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up to date with main

## 🏷️ Commit Message Format

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

## 🤝 Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Provide constructive feedback
- Follow the project's code of conduct

## 📞 Getting Help

- Check existing issues and PRs
- Search documentation
- Ask questions in discussions
- Contact maintainers for urgent issues

Thank you for contributing to the RAG Quote System! 🎉 