"""
Flutter build validation and retry mechanisms.
Validates dependency resolutions by executing actual Flutter commands and provides intelligent retry logic.
"""

import asyncio
import subprocess
import time
import os
import shutil
import tempfile
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import json
import re
from enum import Enum


class BuildStatus(Enum):
    """Build status enumeration."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    DEPENDENCY_ERROR = "dependency_error"
    COMPILATION_ERROR = "compilation_error"
    ANALYSIS_ERROR = "analysis_error"


@dataclass
class BuildResult:
    """Result of a Flutter build operation."""
    status: BuildStatus
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    command: str
    project_path: Path
    error_details: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationConfig:
    """Configuration for build validation."""
    # Timeouts (in seconds)
    pub_get_timeout: int = 300
    build_timeout: int = 600
    analyze_timeout: int = 120
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 5.0
    exponential_backoff: bool = True
    
    # Build targets
    build_targets: List[str] = field(default_factory=lambda: ['android', 'ios'])
    build_modes: List[str] = field(default_factory=lambda: ['debug'])
    
    # Validation steps
    run_pub_get: bool = True
    run_analyze: bool = True
    run_build: bool = True
    run_test: bool = False
    
    # Environment settings
    flutter_channel: str = 'stable'
    use_isolated_environment: bool = True
    preserve_build_artifacts: bool = False
    
    # Performance settings
    max_concurrent_builds: int = 2
    memory_limit_mb: int = 4096
    cpu_limit_percent: int = 80


class FlutterEnvironment:
    """Manages Flutter SDK environment and version detection."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._flutter_path = None
        self._dart_path = None
        self._flutter_version = None
        self._dart_version = None
    
    def detect_flutter_installation(self) -> bool:
        """Detect Flutter installation and validate environment."""
        try:
            # Try to find Flutter in PATH
            result = subprocess.run(['which', 'flutter'], capture_output=True, text=True)
            if result.returncode == 0:
                self._flutter_path = result.stdout.strip()
            else:
                # Try common installation paths
                common_paths = [
                    '/usr/local/bin/flutter',
                    '/opt/flutter/bin/flutter',
                    os.path.expanduser('~/flutter/bin/flutter'),
                    os.path.expanduser('~/development/flutter/bin/flutter')
                ]
                
                for path in common_paths:
                    if os.path.exists(path):
                        self._flutter_path = path
                        break
            
            if not self._flutter_path:
                self.logger.error("Flutter not found in PATH or common locations")
                return False
            
            # Verify Flutter works
            result = subprocess.run([self._flutter_path, '--version'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                self.logger.error(f"Flutter command failed: {result.stderr}")
                return False
            
            # Extract version information
            self._parse_flutter_version(result.stdout)
            
            # Find Dart SDK
            self._detect_dart_sdk()
            
            self.logger.info(f"Flutter detected: {self._flutter_version}")
            self.logger.info(f"Dart detected: {self._dart_version}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to detect Flutter: {e}")
            return False
    
    def _parse_flutter_version(self, version_output: str):
        """Parse Flutter version from command output."""
        lines = version_output.split('\n')
        for line in lines:
            if line.startswith('Flutter'):
                # Extract version number
                match = re.search(r'Flutter (\d+\.\d+\.\d+)', line)
                if match:
                    self._flutter_version = match.group(1)
                break
    
    def _detect_dart_sdk(self):
        """Detect Dart SDK path and version."""
        try:
            # Dart is usually bundled with Flutter
            flutter_dir = Path(self._flutter_path).parent.parent
            dart_path = flutter_dir / 'bin' / 'cache' / 'dart-sdk' / 'bin' / 'dart'
            
            if dart_path.exists():
                self._dart_path = str(dart_path)
            else:
                # Try system Dart
                result = subprocess.run(['which', 'dart'], capture_output=True, text=True)
                if result.returncode == 0:
                    self._dart_path = result.stdout.strip()
            
            if self._dart_path:
                result = subprocess.run([self._dart_path, '--version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    # Parse Dart version
                    match = re.search(r'Dart SDK version: (\d+\.\d+\.\d+)', result.stderr)
                    if match:
                        self._dart_version = match.group(1)
        
        except Exception as e:
            self.logger.warning(f"Failed to detect Dart SDK: {e}")
    
    @property
    def flutter_path(self) -> Optional[str]:
        return self._flutter_path
    
    @property
    def dart_path(self) -> Optional[str]:
        return self._dart_path
    
    @property
    def flutter_version(self) -> Optional[str]:
        return self._flutter_version
    
    @property
    def dart_version(self) -> Optional[str]:
        return self._dart_version


class BuildValidator:
    """Validates Flutter builds with comprehensive error analysis."""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger(__name__)
        self.flutter_env = FlutterEnvironment()
        
        # Initialize Flutter environment
        if not self.flutter_env.detect_flutter_installation():
            raise RuntimeError("Flutter SDK not found or not properly configured")
        
        # Build execution tracking
        self.active_builds = {}
        self.build_history = []
        
        # Performance monitoring
        self.resource_monitor = ResourceMonitor()
    
    async def validate_resolution(self, project_path: Path, 
                                resolution: Dict[str, str] = None) -> BuildResult:
        """Validate a dependency resolution by running Flutter commands."""
        start_time = time.time()
        
        try:
            # Create isolated environment if requested
            if self.config.use_isolated_environment:
                with self._create_isolated_environment(project_path) as isolated_path:
                    return await self._run_validation_steps(isolated_path, resolution)
            else:
                return await self._run_validation_steps(project_path, resolution)
        
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return BuildResult(
                status=BuildStatus.FAILURE,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=time.time() - start_time,
                command="validation",
                project_path=project_path,
                error_details={'exception': str(e)}
            )
    
    def _create_isolated_environment(self, project_path: Path):
        """Create isolated environment for testing."""
        return IsolatedEnvironment(project_path, self.config)
    
    async def _run_validation_steps(self, project_path: Path, 
                                  resolution: Dict[str, str] = None) -> BuildResult:
        """Run all validation steps in sequence."""
        results = []
        
        # Step 1: flutter pub get
        if self.config.run_pub_get:
            pub_get_result = await self._run_pub_get(project_path)
            results.append(pub_get_result)
            
            if pub_get_result.status != BuildStatus.SUCCESS:
                return pub_get_result
        
        # Step 2: flutter analyze
        if self.config.run_analyze:
            analyze_result = await self._run_analyze(project_path)
            results.append(analyze_result)
            
            # Continue even if analyze has warnings, but fail on errors
            if analyze_result.status == BuildStatus.FAILURE:
                return analyze_result
        
        # Step 3: flutter build
        if self.config.run_build:
            for target in self.config.build_targets:
                for mode in self.config.build_modes:
                    build_result = await self._run_build(project_path, target, mode)
                    results.append(build_result)
                    
                    if build_result.status != BuildStatus.SUCCESS:
                        return build_result
        
        # Step 4: flutter test (optional)
        if self.config.run_test:
            test_result = await self._run_test(project_path)
            results.append(test_result)
            
            if test_result.status != BuildStatus.SUCCESS:
                return test_result
        
        # Combine results
        return self._combine_results(results)
    
    async def _run_pub_get(self, project_path: Path) -> BuildResult:
        """Run flutter pub get command."""
        command = [self.flutter_env.flutter_path, 'pub', 'get']
        
        return await self._execute_flutter_command(
            command, project_path, self.config.pub_get_timeout, "pub_get"
        )
    
    async def _run_analyze(self, project_path: Path) -> BuildResult:
        """Run flutter analyze command."""
        command = [self.flutter_env.flutter_path, 'analyze']
        
        result = await self._execute_flutter_command(
            command, project_path, self.config.analyze_timeout, "analyze"
        )
        
        # Parse analyze output for warnings and errors
        if result.status == BuildStatus.SUCCESS:
            warnings = self._parse_analyze_warnings(result.stdout)
            result.warnings.extend(warnings)
        
        return result
    
    async def _run_build(self, project_path: Path, target: str, mode: str) -> BuildResult:
        """Run flutter build command for specific target and mode."""
        command = [self.flutter_env.flutter_path, 'build', target, f'--{mode}']
        
        # Add target-specific flags
        if target == 'web':
            command.extend(['--web-renderer', 'html'])
        elif target == 'ios':
            command.extend(['--no-codesign'])
        
        return await self._execute_flutter_command(
            command, project_path, self.config.build_timeout, f"build_{target}_{mode}"
        )
    
    async def _run_test(self, project_path: Path) -> BuildResult:
        """Run flutter test command."""
        command = [self.flutter_env.flutter_path, 'test']
        
        return await self._execute_flutter_command(
            command, project_path, self.config.build_timeout, "test"
        )
    
    async def _execute_flutter_command(self, command: List[str], project_path: Path,
                                     timeout: int, operation: str) -> BuildResult:
        """Execute a Flutter command with monitoring and error handling."""
        start_time = time.time()
        command_str = ' '.join(command)
        
        self.logger.info(f"Executing: {command_str} in {project_path}")
        
        try:
            # Start resource monitoring
            monitor_task = asyncio.create_task(
                self.resource_monitor.monitor_process(operation)
            )
            
            # Execute command
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._get_flutter_environment()
            )
            
            # Track active build
            self.active_builds[process.pid] = {
                'command': command_str,
                'start_time': start_time,
                'operation': operation
            }
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                
                execution_time = time.time() - start_time
                
                # Stop monitoring
                monitor_task.cancel()
                
                # Clean up tracking
                if process.pid in self.active_builds:
                    del self.active_builds[process.pid]
                
                # Decode output
                stdout_str = stdout.decode('utf-8', errors='replace')
                stderr_str = stderr.decode('utf-8', errors='replace')
                
                # Determine status
                status = BuildStatus.SUCCESS if process.returncode == 0 else BuildStatus.FAILURE
                
                # Analyze errors
                error_details = None
                if status == BuildStatus.FAILURE:
                    error_details = self._analyze_build_error(stdout_str, stderr_str, operation)
                    status = error_details.get('status', BuildStatus.FAILURE)
                
                result = BuildResult(
                    status=status,
                    exit_code=process.returncode,
                    stdout=stdout_str,
                    stderr=stderr_str,
                    execution_time=execution_time,
                    command=command_str,
                    project_path=project_path,
                    error_details=error_details
                )
                
                # Add to history
                self.build_history.append(result)
                
                return result
                
            except asyncio.TimeoutError:
                # Kill the process
                try:
                    process.kill()
                    await process.wait()
                except:
                    pass
                
                # Clean up
                monitor_task.cancel()
                if process.pid in self.active_builds:
                    del self.active_builds[process.pid]
                
                return BuildResult(
                    status=BuildStatus.TIMEOUT,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Command timed out after {timeout} seconds",
                    execution_time=time.time() - start_time,
                    command=command_str,
                    project_path=project_path,
                    error_details={'timeout': timeout}
                )
        
        except Exception as e:
            return BuildResult(
                status=BuildStatus.FAILURE,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=time.time() - start_time,
                command=command_str,
                project_path=project_path,
                error_details={'exception': str(e)}
            )
    
    def _get_flutter_environment(self) -> Dict[str, str]:
        """Get environment variables for Flutter commands."""
        env = os.environ.copy()
        
        # Add Flutter-specific environment variables
        env['FLUTTER_ROOT'] = str(Path(self.flutter_env.flutter_path).parent.parent)
        env['PUB_CACHE'] = env.get('PUB_CACHE', os.path.expanduser('~/.pub-cache'))
        
        # Disable analytics and crash reporting for CI/automated environments
        env['FLUTTER_ANALYTICS'] = 'false'
        env['DART_ANALYTICS'] = 'false'
        
        return env
    
    def _analyze_build_error(self, stdout: str, stderr: str, operation: str) -> Dict[str, Any]:
        """Analyze build errors to categorize and extract useful information."""
        error_details = {
            'operation': operation,
            'category': 'unknown',
            'suggestions': []
        }
        
        combined_output = stdout + stderr
        
        # Dependency-related errors
        if any(keyword in combined_output.lower() for keyword in [
            'version solving failed', 'dependency conflict', 'incompatible dependencies',
            'could not find a version', 'version constraint'
        ]):
            error_details['category'] = 'dependency'
            error_details['status'] = BuildStatus.DEPENDENCY_ERROR
            error_details['suggestions'].append('Try updating dependency constraints')
            error_details['suggestions'].append('Check for conflicting version requirements')
        
        # Compilation errors
        elif any(keyword in combined_output.lower() for keyword in [
            'compilation error', 'syntax error', 'undefined name', 'type error'
        ]):
            error_details['category'] = 'compilation'
            error_details['status'] = BuildStatus.COMPILATION_ERROR
            error_details['suggestions'].append('Fix compilation errors in source code')
        
        # Analysis errors
        elif 'analyze' in operation and any(keyword in combined_output.lower() for keyword in [
            'analysis error', 'lint error', 'static analysis'
        ]):
            error_details['category'] = 'analysis'
            error_details['status'] = BuildStatus.ANALYSIS_ERROR
            error_details['suggestions'].append('Fix static analysis issues')
        
        # Network/connectivity errors
        elif any(keyword in combined_output.lower() for keyword in [
            'network error', 'connection failed', 'timeout', 'dns resolution'
        ]):
            error_details['category'] = 'network'
            error_details['suggestions'].append('Check internet connectivity')
            error_details['suggestions'].append('Try again later')
        
        # Extract specific error messages
        error_lines = []
        for line in combined_output.split('\n'):
            if any(keyword in line.lower() for keyword in ['error:', 'failed:', 'exception:']):
                error_lines.append(line.strip())
        
        error_details['error_messages'] = error_lines[:10]  # Limit to first 10 errors
        
        return error_details
    
    def _parse_analyze_warnings(self, analyze_output: str) -> List[str]:
        """Parse warnings from flutter analyze output."""
        warnings = []
        
        for line in analyze_output.split('\n'):
            if 'warning:' in line.lower() or 'info:' in line.lower():
                warnings.append(line.strip())
        
        return warnings
    
    def _combine_results(self, results: List[BuildResult]) -> BuildResult:
        """Combine multiple build results into a single result."""
        if not results:
            return BuildResult(
                status=BuildStatus.FAILURE,
                exit_code=-1,
                stdout="",
                stderr="No results to combine",
                execution_time=0.0,
                command="combined",
                project_path=Path(".")
            )
        
        # Overall status is success only if all steps succeeded
        overall_status = BuildStatus.SUCCESS
        for result in results:
            if result.status != BuildStatus.SUCCESS:
                overall_status = result.status
                break
        
        # Combine outputs
        combined_stdout = '\n'.join(result.stdout for result in results)
        combined_stderr = '\n'.join(result.stderr for result in results)
        
        # Sum execution times
        total_time = sum(result.execution_time for result in results)
        
        # Combine warnings
        all_warnings = []
        for result in results:
            all_warnings.extend(result.warnings)
        
        return BuildResult(
            status=overall_status,
            exit_code=results[-1].exit_code,
            stdout=combined_stdout,
            stderr=combined_stderr,
            execution_time=total_time,
            command="combined_validation",
            project_path=results[0].project_path,
            warnings=all_warnings
        )


class IsolatedEnvironment:
    """Context manager for isolated build environments."""
    
    def __init__(self, project_path: Path, config: ValidationConfig):
        self.project_path = project_path
        self.config = config
        self.temp_dir = None
        self.isolated_path = None
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self) -> Path:
        """Create isolated environment."""
        try:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix='flutter_validation_')
            self.isolated_path = Path(self.temp_dir) / 'project'
            
            # Copy project to isolated environment
            shutil.copytree(self.project_path, self.isolated_path, 
                          ignore=shutil.ignore_patterns('.dart_tool', 'build', '.packages'))
            
            self.logger.info(f"Created isolated environment at {self.isolated_path}")
            return self.isolated_path
            
        except Exception as e:
            self.logger.error(f"Failed to create isolated environment: {e}")
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up isolated environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                if not self.config.preserve_build_artifacts:
                    shutil.rmtree(self.temp_dir, ignore_errors=True)
                    self.logger.info(f"Cleaned up isolated environment")
                else:
                    self.logger.info(f"Preserved isolated environment at {self.temp_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up isolated environment: {e}")


class ResourceMonitor:
    """Monitors system resources during builds."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def monitor_process(self, operation: str):
        """Monitor system resources during operation."""
        start_time = time.time()
        max_memory = 0
        max_cpu = 0
        
        try:
            while True:
                # Get current system stats
                memory_percent = psutil.virtual_memory().percent
                cpu_percent = psutil.cpu_percent(interval=1)
                
                max_memory = max(max_memory, memory_percent)
                max_cpu = max(max_cpu, cpu_percent)
                
                # Log if resources are high
                if memory_percent > 90 or cpu_percent > 95:
                    self.logger.warning(
                        f"High resource usage during {operation}: "
                        f"Memory: {memory_percent:.1f}%, CPU: {cpu_percent:.1f}%"
                    )
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
        except asyncio.CancelledError:
            duration = time.time() - start_time
            self.logger.info(
                f"Resource monitoring for {operation} completed. "
                f"Duration: {duration:.1f}s, Max Memory: {max_memory:.1f}%, Max CPU: {max_cpu:.1f}%"
            )


class RetryManager:
    """Manages retry logic for failed builds."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def retry_with_backoff(self, operation_func, *args, **kwargs) -> BuildResult:
        """Retry operation with exponential backoff."""
        last_result = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                result = await operation_func(*args, **kwargs)
                
                if result.status == BuildStatus.SUCCESS:
                    if attempt > 0:
                        self.logger.info(f"Operation succeeded on attempt {attempt + 1}")
                    return result
                
                last_result = result
                
                # Don't retry certain types of errors
                if self._should_not_retry(result):
                    self.logger.info(f"Not retrying due to error type: {result.status}")
                    return result
                
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    self.logger.info(f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed with exception: {e}")
                if attempt == self.config.max_retries:
                    raise
        
        return last_result or BuildResult(
            status=BuildStatus.FAILURE,
            exit_code=-1,
            stdout="",
            stderr="All retry attempts failed",
            execution_time=0.0,
            command="retry",
            project_path=Path(".")
        )
    
    def _should_not_retry(self, result: BuildResult) -> bool:
        """Determine if an error should not be retried."""
        # Don't retry compilation errors or analysis errors
        if result.status in [BuildStatus.COMPILATION_ERROR, BuildStatus.ANALYSIS_ERROR]:
            return True
        
        # Don't retry if error details suggest it's not transient
        if result.error_details:
            category = result.error_details.get('category', '')
            if category in ['compilation', 'analysis']:
                return True
        
        return False
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.config.exponential_backoff:
            return self.config.retry_delay * (2 ** attempt)
        else:
            return self.config.retry_delay


def setup_validation_logging():
    """Setup logging for validation components."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


if __name__ == "__main__":
    setup_validation_logging()
    
    # Example usage
    config = ValidationConfig()
    validator = BuildValidator(config)
    
    print("Flutter build validator initialized successfully")
    print(f"Flutter version: {validator.flutter_env.flutter_version}")
    print(f"Dart version: {validator.flutter_env.dart_version}")
    print("Ready to validate builds!")

