# ML-Powered Flutter Dependency Resolver

A sophisticated machine learning-powered system that automatically resolves versioning issues in Flutter projects' `pubspec.yaml` files, ensures `flutter pub get` completes successfully, and guarantees successful project builds.

## 🚀 Features

- **Intelligent Dependency Analysis**: Advanced parsing and analysis of Flutter project dependencies
- **Machine Learning Resolution**: Uses reinforcement learning and graph neural networks to find optimal dependency versions
- **Automatic Conflict Resolution**: Detects and resolves version conflicts across multiple `pubspec.yaml` files
- **Build Validation**: Validates resolutions by running actual Flutter commands (`pub get`, `analyze`, `build`)
- **Multi-Project Support**: Handles multiple Flutter projects simultaneously
- **Comprehensive Retry Logic**: Intelligent retry mechanisms with exponential backoff
- **Detailed Reporting**: Generates comprehensive reports of resolution processes
- **Preservation of Formatting**: Maintains original `pubspec.yaml` formatting and comments

## 🏗️ Architecture

The system consists of several integrated components:

1. **Dependency Analyzer** (`src/analysis/`): Parses and analyzes Flutter project dependencies
2. **ML Core** (`src/ml/`): Machine learning algorithms for dependency resolution
3. **Pubspec Engine** (`src/core/`): Advanced `pubspec.yaml` parsing and modification
4. **Version Resolver** (`src/ml/version_resolver.py`): ML-powered version resolution algorithms
5. **Build Validator** (`src/validation/`): Flutter build validation and testing
6. **Main Integration** (`main.py`): Unified interface combining all components

## 📋 Requirements

### System Requirements
- Python 3.11+
- Flutter SDK 3.0+
- Dart SDK 3.0+
- 4GB+ RAM (8GB recommended for large projects)
- Internet connection (for package information retrieval)

### Python Dependencies
```
torch>=2.0.0
scikit-learn>=1.3.0
networkx>=3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
ruamel.yaml>=0.18.0
semantic-version>=2.10.0
packaging>=23.0
aiohttp>=3.8.0
psutil>=5.9.0
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd flutter_ml_dependency_resolver
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify Flutter installation**:
```bash
flutter --version
```

## 🚀 Quick Start

### Basic Usage

Resolve dependencies for a single Flutter project:
```bash
python main.py /path/to/flutter/project
```

### Dry Run (Analysis Only)

Analyze dependencies without making changes:
```bash
python main.py /path/to/flutter/project --dry-run
```

### Multiple Projects

Resolve dependencies for multiple projects:
```bash
python main.py /path/to/project1 /path/to/project2 /path/to/project3
```

### Custom Configuration

Use a custom configuration file:
```bash
python main.py /path/to/project --config config.json
```

### Generate Report

Save a detailed resolution report:
```bash
python main.py /path/to/project --report resolution_report.json
```

## ⚙️ Configuration

Create a `config.json` file to customize behavior:

```json
{
  "validation": {
    "pub_get_timeout": 300,
    "build_timeout": 600,
    "analyze_timeout": 120,
    "max_retries": 3,
    "build_targets": ["android", "ios"],
    "build_modes": ["debug"],
    "run_pub_get": true,
    "run_analyze": true,
    "run_build": true,
    "use_isolated_environment": true
  },
  "ml": {
    "max_candidates": 10,
    "optimization_goals": {
      "stability": 0.4,
      "compatibility": 0.3,
      "security": 0.2,
      "performance": 0.1
    }
  },
  "logging": {
    "level": "INFO",
    "file": "resolver.log"
  }
}
```

## 📊 How It Works

### 1. Dependency Analysis
- Parses all `pubspec.yaml` files in the project
- Builds dependency graphs with constraints
- Identifies conflicts and incompatibilities

### 2. ML-Powered Resolution
- Uses reinforcement learning to explore version combinations
- Employs graph neural networks to understand dependency relationships
- Generates multiple candidate resolutions with confidence scores

### 3. Constraint Solving
- Applies semantic versioning rules
- Resolves transitive dependencies
- Ensures compatibility across all packages

### 4. Build Validation
- Creates isolated test environments
- Runs `flutter pub get`, `flutter analyze`, and `flutter build`
- Validates that the resolution actually works

### 5. Intelligent Retry
- Automatically retries failed operations
- Uses exponential backoff for network issues
- Learns from failures to improve future resolutions

## 🧪 Testing

Run the comprehensive test suite:
```bash
python tests/test_framework.py
```

Test with example projects:
```bash
# Simple project
python main.py examples/simple_app --dry-run

# Complex project with many dependencies
python main.py examples/complex_app --dry-run

# Project with intentional conflicts
python main.py examples/conflicted_app --dry-run
```

## 📈 Performance

The system is optimized for performance:

- **Concurrent Processing**: Handles multiple projects simultaneously
- **Caching**: Caches package information to reduce API calls
- **Incremental Analysis**: Only re-analyzes changed dependencies
- **Resource Monitoring**: Monitors CPU and memory usage during builds

Typical performance metrics:
- Simple project (5-10 dependencies): 10-30 seconds
- Complex project (20+ dependencies): 1-3 minutes
- Large enterprise project (50+ dependencies): 3-10 minutes

## 🔧 Advanced Usage

### Custom ML Configuration

Adjust machine learning parameters:
```python
from src.ml.ml_core import MLConfig

config = MLConfig(
    state_dim=256,
    action_dim=200,
    learning_rate=0.001,
    batch_size=64,
    memory_size=20000
)
```

### Programmatic API

Use the resolver programmatically:
```python
import asyncio
from pathlib import Path
from main import FlutterDependencyResolver

async def resolve_project():
    resolver = FlutterDependencyResolver()
    result = await resolver.resolve_project(Path('/path/to/project'))
    print(f"Resolution successful: {result['success']}")

asyncio.run(resolve_project())
```

### Custom Validation Rules

Extend the build validator:
```python
from src.validation.build_validator import BuildValidator, ValidationConfig

config = ValidationConfig(
    build_targets=['android', 'ios', 'web'],
    build_modes=['debug', 'release'],
    run_test=True
)

validator = BuildValidator(config)
```

## 🐛 Troubleshooting

### Common Issues

1. **Flutter not found**:
   - Ensure Flutter is in your PATH
   - Check with `flutter --version`

2. **Permission errors**:
   - Run with appropriate permissions
   - Check file system permissions

3. **Network timeouts**:
   - Increase timeout values in configuration
   - Check internet connectivity

4. **Memory issues**:
   - Reduce `max_concurrent` parameter
   - Increase system memory

### Debug Mode

Enable verbose logging:
```bash
python main.py /path/to/project --verbose
```

### Log Analysis

Check logs for detailed error information:
```bash
tail -f resolver.log
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Flutter team for the excellent framework
- PyTorch team for machine learning capabilities
- The open-source community for various dependencies

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the example projects

---

**Note**: This tool modifies your `pubspec.yaml` files. Always use version control and test thoroughly before deploying to production.

