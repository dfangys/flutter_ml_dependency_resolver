# User Guide: ML-Powered Flutter Dependency Resolver

## Table of Contents
1. [Getting Started](#getting-started)
2. [Basic Operations](#basic-operations)
3. [Advanced Features](#advanced-features)
4. [Configuration Guide](#configuration-guide)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)

## Getting Started

### Prerequisites

Before using the ML-powered Flutter dependency resolver, ensure you have:

1. **Flutter SDK**: Version 3.0 or higher
2. **Python**: Version 3.11 or higher
3. **System Resources**: At least 4GB RAM (8GB recommended)
4. **Internet Connection**: Required for package information retrieval

### Installation Steps

1. **Download and Extract**:
   ```bash
   # Extract the resolver to your preferred location
   cd /path/to/flutter_ml_dependency_resolver
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**:
   ```bash
   python main.py --help
   ```

### First Run

Test the resolver with the provided examples:

```bash
# Analyze a simple project (dry run)
python main.py examples/simple_app --dry-run

# View the analysis results
cat resolution_report.json
```

## Basic Operations

### Single Project Resolution

**Syntax**: `python main.py <project_path> [options]`

**Examples**:

```bash
# Basic resolution
python main.py /path/to/my_flutter_app

# Dry run (analysis only)
python main.py /path/to/my_flutter_app --dry-run

# With custom configuration
python main.py /path/to/my_flutter_app --config my_config.json

# Generate detailed report
python main.py /path/to/my_flutter_app --report detailed_report.json
```

### Multiple Project Resolution

Process multiple Flutter projects simultaneously:

```bash
# Resolve multiple projects
python main.py /path/to/app1 /path/to/app2 /path/to/app3

# Limit concurrent processing
python main.py /path/to/app1 /path/to/app2 --max-concurrent 2

# Batch processing with reporting
python main.py /path/to/projects/* --report batch_report.json
```

### Understanding Output

The resolver provides detailed output including:

1. **Analysis Phase**:
   - Dependency graph construction
   - Conflict detection
   - Constraint analysis

2. **Resolution Phase**:
   - ML candidate generation
   - Scoring and ranking
   - Best solution selection

3. **Validation Phase**:
   - Build testing
   - Error detection
   - Success confirmation

**Example Output**:
```
2024-01-15 10:30:15 - Starting dependency resolution for /path/to/project
2024-01-15 10:30:16 - Analyzing current dependencies...
2024-01-15 10:30:17 - Found 3 dependency conflicts
2024-01-15 10:30:18 - Generating ML-powered resolution...
2024-01-15 10:30:25 - Applying dependency resolution...
2024-01-15 10:30:26 - Validating resolution...
2024-01-15 10:30:45 - Resolution completed in 30.2s

RESOLUTION SUMMARY
==================
Projects processed: 1
Successful: 1
Failed: 0
Average execution time: 30.20s
```

## Advanced Features

### Custom Optimization Goals

Adjust the ML algorithm's optimization priorities:

```json
{
  "ml": {
    "optimization_goals": {
      "stability": 0.5,     // Prefer stable versions
      "compatibility": 0.3, // Ensure compatibility
      "security": 0.15,     // Consider security updates
      "performance": 0.05   // Optimize for performance
    }
  }
}
```

### Build Target Configuration

Specify which platforms to validate:

```json
{
  "validation": {
    "build_targets": ["android", "ios", "web"],
    "build_modes": ["debug", "release"],
    "run_test": true
  }
}
```

### Timeout and Retry Configuration

Customize timeout and retry behavior:

```json
{
  "validation": {
    "pub_get_timeout": 600,    // 10 minutes for pub get
    "build_timeout": 1200,     // 20 minutes for build
    "max_retries": 5,          // Retry up to 5 times
    "retry_delay": 10.0,       // 10 second base delay
    "exponential_backoff": true
  }
}
```

### Isolated Environment Testing

Enable isolated testing to prevent interference:

```json
{
  "validation": {
    "use_isolated_environment": true,
    "preserve_build_artifacts": false
  }
}
```

## Configuration Guide

### Complete Configuration Example

```json
{
  "validation": {
    "pub_get_timeout": 300,
    "build_timeout": 600,
    "analyze_timeout": 120,
    "max_retries": 3,
    "retry_delay": 5.0,
    "exponential_backoff": true,
    "build_targets": ["android", "ios"],
    "build_modes": ["debug"],
    "run_pub_get": true,
    "run_analyze": true,
    "run_build": true,
    "run_test": false,
    "flutter_channel": "stable",
    "use_isolated_environment": true,
    "preserve_build_artifacts": false,
    "max_concurrent_builds": 2,
    "memory_limit_mb": 4096,
    "cpu_limit_percent": 80
  },
  "ml": {
    "max_candidates": 10,
    "optimization_goals": {
      "stability": 0.4,
      "performance": 0.3,
      "security": 0.2,
      "compatibility": 0.1
    },
    "state_dim": 128,
    "action_dim": 100,
    "learning_rate": 0.001,
    "batch_size": 32,
    "memory_size": 10000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "target_update": 10,
    "gamma": 0.99
  },
  "logging": {
    "level": "INFO",
    "file": "resolver.log",
    "max_file_size": "10MB",
    "backup_count": 5
  }
}
```

### Environment Variables

Set environment variables for additional configuration:

```bash
export FLUTTER_ML_RESOLVER_CONFIG="/path/to/config.json"
export FLUTTER_ML_RESOLVER_LOG_LEVEL="DEBUG"
export FLUTTER_ML_RESOLVER_CACHE_DIR="/tmp/flutter_ml_cache"
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Flutter Not Found

**Problem**: `Flutter SDK not found or not properly configured`

**Solutions**:
- Verify Flutter installation: `flutter --version`
- Add Flutter to PATH: `export PATH="$PATH:/path/to/flutter/bin"`
- Check Flutter doctor: `flutter doctor`

#### 2. Permission Errors

**Problem**: `Permission denied when modifying pubspec.yaml`

**Solutions**:
- Check file permissions: `ls -la pubspec.yaml`
- Run with appropriate permissions
- Ensure the project directory is writable

#### 3. Network Timeouts

**Problem**: `Failed to fetch package info: timeout`

**Solutions**:
- Increase timeout values in configuration
- Check internet connectivity
- Use a different DNS server
- Check firewall settings

#### 4. Memory Issues

**Problem**: `Out of memory during resolution`

**Solutions**:
- Reduce `max_concurrent` parameter
- Increase system memory
- Close other applications
- Use smaller batch sizes

#### 5. Build Failures

**Problem**: `Flutter build failed after resolution`

**Solutions**:
- Check Flutter doctor: `flutter doctor`
- Clean the project: `flutter clean`
- Update Flutter: `flutter upgrade`
- Check platform-specific requirements

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Verbose output
python main.py /path/to/project --verbose

# Quiet mode (errors only)
python main.py /path/to/project --quiet

# Custom log file
python main.py /path/to/project --config config_with_logging.json
```

### Log Analysis

Analyze logs for issues:

```bash
# View recent logs
tail -f resolver.log

# Search for errors
grep -i error resolver.log

# Filter by component
grep "DependencyAnalyzer" resolver.log
```

## Best Practices

### 1. Version Control

Always use version control before running the resolver:

```bash
# Commit current state
git add .
git commit -m "Before dependency resolution"

# Run resolver
python main.py /path/to/project

# Review changes
git diff

# Commit if satisfied
git add .
git commit -m "Applied ML dependency resolution"
```

### 2. Testing Strategy

Follow a systematic testing approach:

1. **Dry Run First**: Always start with `--dry-run`
2. **Isolated Testing**: Use isolated environments
3. **Incremental Validation**: Test one project at a time
4. **Full Build Testing**: Validate complete build process

```bash
# Step 1: Dry run analysis
python main.py /path/to/project --dry-run

# Step 2: Apply resolution
python main.py /path/to/project

# Step 3: Manual verification
cd /path/to/project
flutter pub get
flutter analyze
flutter build android
```

### 3. Configuration Management

Maintain project-specific configurations:

```bash
# Project-specific config
project_root/
├── pubspec.yaml
├── flutter_resolver_config.json
└── lib/
```

### 4. Batch Processing

For multiple projects, use systematic batch processing:

```bash
# Create project list
find /path/to/projects -name "pubspec.yaml" -exec dirname {} \; > project_list.txt

# Process in batches
while read project; do
  echo "Processing $project"
  python main.py "$project" --config batch_config.json
done < project_list.txt
```

### 5. Monitoring and Reporting

Implement monitoring for production use:

```bash
# Generate comprehensive reports
python main.py /path/to/projects/* --report "$(date +%Y%m%d)_resolution_report.json"

# Monitor system resources
top -p $(pgrep -f "main.py")

# Track resolution success rates
grep "successful" resolver.log | wc -l
```

### 6. Backup and Recovery

Implement backup strategies:

```bash
# Backup before resolution
cp pubspec.yaml pubspec.yaml.backup

# Automated backup script
#!/bin/bash
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
find . -name "pubspec.yaml" -exec cp {} "$BACKUP_DIR/" \;
```

### 7. Performance Optimization

Optimize for your environment:

1. **Adjust Concurrency**: Match your system capabilities
2. **Cache Management**: Use persistent caching for repeated runs
3. **Resource Limits**: Set appropriate memory and CPU limits
4. **Network Optimization**: Use local package mirrors if available

```json
{
  "validation": {
    "max_concurrent_builds": 4,  // Adjust based on CPU cores
    "memory_limit_mb": 8192,     // Adjust based on available RAM
    "use_isolated_environment": false  // Disable for faster processing
  }
}
```

---

This user guide provides comprehensive information for effectively using the ML-powered Flutter dependency resolver. For additional support, refer to the API documentation and example projects.

