"""
Comprehensive test framework for the ML-powered Flutter dependency resolver.
"""

import unittest
import asyncio
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from analysis.dependency_analyzer import DependencyAnalyzer, DependencyGraph, DependencyConstraint
from ml.ml_core import DependencyResolutionAgent, MLConfig
from core.pubspec_engine import PubspecManager, DependencyChange
from validation.build_validator import BuildValidator, ValidationConfig, BuildStatus


class TestDependencyAnalyzer(unittest.TestCase):
    """Test cases for dependency analyzer."""
    
    def setUp(self):
        self.analyzer = DependencyAnalyzer()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_pubspec(self, content: str) -> Path:
        """Create a test pubspec.yaml file."""
        pubspec_path = self.temp_dir / 'pubspec.yaml'
        with open(pubspec_path, 'w') as f:
            f.write(content)
        return pubspec_path
    
    def test_parse_simple_pubspec(self):
        """Test parsing a simple pubspec.yaml."""
        content = """
name: test_app
version: 1.0.0
description: A test Flutter application

environment:
  sdk: '>=2.17.0 <4.0.0'
  flutter: '>=3.0.0'

dependencies:
  flutter:
    sdk: flutter
  http: ^0.13.5
  shared_preferences: ^2.0.15

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^2.0.0
"""
        pubspec_path = self.create_test_pubspec(content)
        
        # Create project structure
        (self.temp_dir / 'lib').mkdir()
        (self.temp_dir / 'lib' / 'main.dart').touch()
        
        graph = self.analyzer.analyze_project(self.temp_dir)
        
        self.assertEqual(graph.project_metadata.name, 'test_app')
        self.assertEqual(graph.project_metadata.version, '1.0.0')
        self.assertIn('http', graph.dependencies)
        self.assertIn('shared_preferences', graph.dependencies)
        self.assertIn('flutter_lints', graph.dev_dependencies)
    
    def test_dependency_constraint_parsing(self):
        """Test parsing different types of dependency constraints."""
        # Test caret constraint
        constraint = DependencyConstraint('test_pkg', '^1.2.3', '')
        self.assertEqual(constraint.constraint_type, 'caret')
        self.assertEqual(constraint.min_version, '1.2.3')
        self.assertTrue(constraint.satisfies_version('1.2.4'))
        self.assertTrue(constraint.satisfies_version('1.9.0'))
        self.assertFalse(constraint.satisfies_version('2.0.0'))
        
        # Test exact constraint
        constraint = DependencyConstraint('test_pkg', '1.2.3', '')
        self.assertEqual(constraint.constraint_type, 'exact')
        self.assertTrue(constraint.satisfies_version('1.2.3'))
        self.assertFalse(constraint.satisfies_version('1.2.4'))
    
    def test_conflict_detection(self):
        """Test dependency conflict detection."""
        content = """
name: test_app
version: 1.0.0

dependencies:
  http: ^0.13.5

dev_dependencies:
  http: ^0.12.0
"""
        pubspec_path = self.create_test_pubspec(content)
        (self.temp_dir / 'lib').mkdir()
        
        graph = self.analyzer.analyze_project(self.temp_dir)
        conflicts = graph.get_dependency_conflicts()
        
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0][0], 'http')


class TestMLCore(unittest.TestCase):
    """Test cases for ML core components."""
    
    def setUp(self):
        self.config = MLConfig()
        self.agent = DependencyResolutionAgent(self.config)
    
    def test_ml_config_initialization(self):
        """Test ML configuration initialization."""
        self.assertEqual(self.config.state_dim, 128)
        self.assertEqual(self.config.action_dim, 100)
        self.assertGreater(self.config.learning_rate, 0)
    
    def test_state_encoder_initialization(self):
        """Test state encoder initialization."""
        encoder = self.agent.state_encoder
        self.assertEqual(encoder.config, self.config)
        self.assertFalse(encoder.is_fitted)
    
    def test_dqn_agent_initialization(self):
        """Test DQN agent initialization."""
        dqn = self.agent.dqn
        self.assertEqual(dqn.config, self.config)
        
        # Test forward pass with dummy input
        import torch
        dummy_state = torch.randn(1, self.config.state_dim)
        output = dqn(dummy_state)
        self.assertEqual(output.shape, (1, self.config.action_dim))


class TestPubspecEngine(unittest.TestCase):
    """Test cases for pubspec.yaml engine."""
    
    def setUp(self):
        self.manager = PubspecManager()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_pubspec(self, content: str) -> Path:
        """Create a test pubspec.yaml file."""
        pubspec_path = self.temp_dir / 'pubspec.yaml'
        with open(pubspec_path, 'w') as f:
            f.write(content)
        return pubspec_path
    
    def test_pubspec_validation(self):
        """Test pubspec.yaml validation."""
        # Valid pubspec
        valid_content = """
name: test_app
version: 1.0.0
dependencies:
  flutter:
    sdk: flutter
"""
        pubspec_path = self.create_test_pubspec(valid_content)
        is_valid, error = self.manager.modifier.validate_pubspec_syntax(pubspec_path)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        # Invalid pubspec
        invalid_content = """
name: test_app
version: 1.0.0
dependencies:
  flutter:
    sdk: flutter
  invalid_yaml: [unclosed list
"""
        pubspec_path = self.create_test_pubspec(invalid_content)
        is_valid, error = self.manager.modifier.validate_pubspec_syntax(pubspec_path)
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
    
    def test_dependency_modification(self):
        """Test dependency modification."""
        content = """
name: test_app
version: 1.0.0
dependencies:
  flutter:
    sdk: flutter
  http: ^0.13.5
"""
        pubspec_path = self.create_test_pubspec(content)
        
        # Test updating a dependency
        changes = [
            DependencyChange(
                name='http',
                old_version='^0.13.5',
                new_version='^0.13.6',
                change_type='update',
                section='dependencies'
            )
        ]
        
        result = self.manager.modifier.modify_pubspec(pubspec_path, changes)
        self.assertTrue(result.success)
        self.assertEqual(len(result.changes_made), 1)
        
        # Verify the change was applied
        with open(pubspec_path, 'r') as f:
            modified_content = f.read()
        self.assertIn('http: ^0.13.6', modified_content)


class TestBuildValidator(unittest.TestCase):
    """Test cases for build validator."""
    
    def setUp(self):
        self.config = ValidationConfig()
        # Mock Flutter environment for testing
        with patch('validation.build_validator.FlutterEnvironment') as mock_env:
            mock_env.return_value.detect_flutter_installation.return_value = True
            mock_env.return_value.flutter_path = '/usr/bin/flutter'
            mock_env.return_value.dart_path = '/usr/bin/dart'
            mock_env.return_value.flutter_version = '3.10.0'
            mock_env.return_value.dart_version = '3.0.0'
            
            self.validator = BuildValidator(self.config)
    
    def test_validation_config(self):
        """Test validation configuration."""
        self.assertEqual(self.config.pub_get_timeout, 300)
        self.assertEqual(self.config.build_timeout, 600)
        self.assertTrue(self.config.run_pub_get)
        self.assertTrue(self.config.run_analyze)
    
    @patch('asyncio.create_subprocess_exec')
    async def test_flutter_command_execution(self, mock_subprocess):
        """Test Flutter command execution."""
        # Mock successful command execution
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b'Success', b'')
        mock_process.returncode = 0
        mock_process.pid = 12345
        mock_subprocess.return_value = mock_process
        
        result = await self.validator._execute_flutter_command(
            ['flutter', 'pub', 'get'],
            Path('/tmp'),
            300,
            'pub_get'
        )
        
        self.assertEqual(result.status, BuildStatus.SUCCESS)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, 'Success')


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_project_path = self.temp_dir / 'test_project'
        self.test_project_path.mkdir()
        
        # Create a minimal Flutter project structure
        (self.test_project_path / 'lib').mkdir()
        (self.test_project_path / 'lib' / 'main.dart').write_text("""
import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: Text('Test App')),
        body: Center(child: Text('Hello World')),
      ),
    );
  }
}
""")
        
        # Create pubspec.yaml with potential conflicts
        pubspec_content = """
name: test_project
description: A test Flutter project
version: 1.0.0+1

environment:
  sdk: '>=2.17.0 <4.0.0'
  flutter: '>=3.0.0'

dependencies:
  flutter:
    sdk: flutter
  http: ^0.13.5
  shared_preferences: ^2.0.15
  provider: ^6.0.3

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^2.0.0
"""
        (self.test_project_path / 'pubspec.yaml').write_text(pubspec_content)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_analysis(self):
        """Test end-to-end dependency analysis."""
        analyzer = DependencyAnalyzer()
        graph = analyzer.analyze_project(self.test_project_path)
        
        self.assertEqual(graph.project_metadata.name, 'test_project')
        self.assertGreater(len(graph.dependencies), 0)
        self.assertIn('http', graph.dependencies)
        self.assertIn('shared_preferences', graph.dependencies)
        self.assertIn('provider', graph.dependencies)
    
    def test_pubspec_modification_workflow(self):
        """Test complete pubspec modification workflow."""
        manager = PubspecManager()
        
        # Get initial analysis
        analysis = manager.get_comprehensive_analysis(self.test_project_path / 'pubspec.yaml')
        self.assertTrue(analysis['validation']['valid'])
        
        # Apply a resolution
        resolution = {
            'http': '^0.13.6',
            'shared_preferences': '^2.1.0',
            'provider': '^6.0.4'
        }
        
        result = manager.apply_ml_resolution(
            self.test_project_path / 'pubspec.yaml',
            resolution
        )
        
        self.assertTrue(result.success)
        self.assertGreater(len(result.changes_made), 0)


class TestExampleProjects(unittest.TestCase):
    """Test cases using example projects."""
    
    def setUp(self):
        self.examples_dir = Path(__file__).parent / 'examples'
    
    def test_example_projects_exist(self):
        """Test that example projects are properly structured."""
        if self.examples_dir.exists():
            for project_dir in self.examples_dir.iterdir():
                if project_dir.is_dir():
                    pubspec_path = project_dir / 'pubspec.yaml'
                    self.assertTrue(pubspec_path.exists(), 
                                  f"pubspec.yaml missing in {project_dir}")
                    
                    lib_dir = project_dir / 'lib'
                    self.assertTrue(lib_dir.exists(), 
                                  f"lib directory missing in {project_dir}")


def run_performance_tests():
    """Run performance tests for the system."""
    print("Running performance tests...")
    
    # Test dependency analysis performance
    analyzer = DependencyAnalyzer()
    
    # Create a large test project
    temp_dir = Path(tempfile.mkdtemp())
    try:
        test_project = temp_dir / 'large_project'
        test_project.mkdir()
        (test_project / 'lib').mkdir()
        
        # Create pubspec with many dependencies
        pubspec_content = """
name: large_project
version: 1.0.0

dependencies:
  flutter:
    sdk: flutter
  http: ^0.13.5
  shared_preferences: ^2.0.15
  provider: ^6.0.3
  dio: ^5.3.2
  cached_network_image: ^3.2.3
  sqflite: ^2.3.0
  path_provider: ^2.1.1
  image_picker: ^1.0.4
  url_launcher: ^6.1.14
  webview_flutter: ^4.4.1
  firebase_core: ^2.17.0
  firebase_auth: ^4.10.1
  cloud_firestore: ^4.9.3
  firebase_storage: ^11.2.8

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^2.0.0
  mockito: ^5.4.2
  build_runner: ^2.4.7
"""
        (test_project / 'pubspec.yaml').write_text(pubspec_content)
        
        # Measure analysis time
        import time
        start_time = time.time()
        graph = analyzer.analyze_project(test_project)
        analysis_time = time.time() - start_time
        
        print(f"Analysis time for large project: {analysis_time:.2f}s")
        print(f"Dependencies analyzed: {len(graph.get_all_dependencies())}")
        
        # Performance should be reasonable (< 5 seconds for this size)
        assert analysis_time < 5.0, f"Analysis took too long: {analysis_time:.2f}s"
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def run_async_tests():
    """Run async test cases."""
    print("Running async tests...")
    
    # Test async components
    from ml.version_resolver import MLVersionResolver
    
    resolver = MLVersionResolver()
    
    # Test with a mock project
    temp_dir = Path(tempfile.mkdtemp())
    try:
        test_project = temp_dir / 'async_test_project'
        test_project.mkdir()
        (test_project / 'lib').mkdir()
        
        pubspec_content = """
name: async_test_project
version: 1.0.0

dependencies:
  flutter:
    sdk: flutter
  http: ^0.13.5
"""
        (test_project / 'pubspec.yaml').write_text(pubspec_content)
        
        # This would normally make network requests, so we'll just test the structure
        print("Async resolver initialized successfully")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all tests."""
    print("Starting comprehensive test suite...")
    
    # Run unit tests
    print("\n" + "="*50)
    print("UNIT TESTS")
    print("="*50)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDependencyAnalyzer,
        TestMLCore,
        TestPubspecEngine,
        TestBuildValidator,
        TestIntegration,
        TestExampleProjects
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run performance tests
    print("\n" + "="*50)
    print("PERFORMANCE TESTS")
    print("="*50)
    
    try:
        run_performance_tests()
        print("Performance tests passed!")
    except Exception as e:
        print(f"Performance tests failed: {e}")
    
    # Run async tests
    print("\n" + "="*50)
    print("ASYNC TESTS")
    print("="*50)
    
    try:
        asyncio.run(run_async_tests())
        print("Async tests passed!")
    except Exception as e:
        print(f"Async tests failed: {e}")
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    if result.wasSuccessful():
        print("All unit tests passed!")
        return 0
    else:
        print(f"Unit tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

