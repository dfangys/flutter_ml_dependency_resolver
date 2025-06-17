"""
Main integration script for ML-powered Flutter dependency resolution.
Combines all components into a unified system for automatic dependency resolution.
"""

import asyncio
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from dataclasses import asdict

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from analysis.dependency_analyzer import DependencyAnalyzer
from ml.version_resolver import MLVersionResolver
from validation.build_validator import BuildValidator, ValidationConfig
from core.pubspec_engine import PubspecManager


class FlutterDependencyResolver:
    """Main class that orchestrates the ML-powered dependency resolution process."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.dependency_analyzer = DependencyAnalyzer()
        self.ml_resolver = MLVersionResolver()
        self.build_validator = BuildValidator(ValidationConfig(**self.config.get('validation', {})))
        self.pubspec_manager = PubspecManager()
        
        # Statistics tracking
        self.stats = {
            'projects_processed': 0,
            'successful_resolutions': 0,
            'failed_resolutions': 0,
            'total_execution_time': 0.0,
            'dependency_conflicts_resolved': 0
        }
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'validation': {
                'pub_get_timeout': 300,
                'build_timeout': 600,
                'analyze_timeout': 120,
                'max_retries': 3,
                'build_targets': ['android'],
                'build_modes': ['debug'],
                'run_pub_get': True,
                'run_analyze': True,
                'run_build': True,
                'use_isolated_environment': True
            },
            'ml': {
                'max_candidates': 10,
                'optimization_goals': {
                    'stability': 0.4,
                    'compatibility': 0.3,
                    'security': 0.2,
                    'performance': 0.1
                }
            },
            'logging': {
                'level': 'INFO',
                'file': None
            }
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                
                # Merge with defaults
                self._deep_merge(default_config, user_config)
                
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _deep_merge(self, base: Dict, update: Dict):
        """Deep merge two dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    async def resolve_project(self, project_path: Path, 
                            dry_run: bool = False,
                            force_rebuild: bool = False) -> Dict[str, Any]:
        """Resolve dependencies for a single Flutter project."""
        start_time = time.time()
        project_path = Path(project_path).resolve()
        
        self.logger.info(f"Starting dependency resolution for {project_path}")
        
        try:
            # Validate project structure
            if not self._validate_project_structure(project_path):
                return {
                    'success': False,
                    'error': 'Invalid Flutter project structure',
                    'project_path': str(project_path)
                }
            
            # Analyze current dependencies
            self.logger.info("Analyzing current dependencies...")
            dependency_graph = self.dependency_analyzer.analyze_project(project_path)
            
            # Check for existing conflicts
            conflicts = dependency_graph.get_dependency_conflicts()
            self.logger.info(f"Found {len(conflicts)} dependency conflicts")
            
            # Generate ML-powered resolution
            self.logger.info("Generating ML-powered resolution...")
            optimization_goals = self.config['ml']['optimization_goals']
            resolution_result = await self.ml_resolver.resolve_dependencies(
                project_path, optimization_goals
            )
            
            if not resolution_result.success:
                return {
                    'success': False,
                    'error': resolution_result.error_message,
                    'project_path': str(project_path),
                    'execution_time': time.time() - start_time
                }
            
            # Apply resolution if not dry run
            if not dry_run:
                self.logger.info("Applying dependency resolution...")
                apply_success = await self.ml_resolver.apply_resolution(
                    project_path, resolution_result
                )
                
                if not apply_success:
                    return {
                        'success': False,
                        'error': 'Failed to apply dependency resolution',
                        'project_path': str(project_path),
                        'execution_time': time.time() - start_time
                    }
            
            # Validate the resolution
            self.logger.info("Validating resolution...")
            if not dry_run or force_rebuild:
                validation_result = await self.build_validator.validate_resolution(
                    project_path, resolution_result.best_candidate.resolution
                )
            else:
                validation_result = None
            
            execution_time = time.time() - start_time
            
            # Update statistics
            self.stats['projects_processed'] += 1
            self.stats['total_execution_time'] += execution_time
            
            if resolution_result.success and (not validation_result or validation_result.status.value == 'success'):
                self.stats['successful_resolutions'] += 1
                self.stats['dependency_conflicts_resolved'] += len(conflicts)
            else:
                self.stats['failed_resolutions'] += 1
            
            # Prepare result
            result = {
                'success': resolution_result.success,
                'project_path': str(project_path),
                'execution_time': execution_time,
                'original_conflicts': len(conflicts),
                'resolution': {
                    'candidate_count': len(resolution_result.all_candidates),
                    'best_score': resolution_result.best_candidate.score if resolution_result.best_candidate else 0,
                    'dependencies': resolution_result.best_candidate.resolution if resolution_result.best_candidate else {}
                },
                'validation': {
                    'status': validation_result.status.value if validation_result else 'skipped',
                    'execution_time': validation_result.execution_time if validation_result else 0,
                    'warnings': validation_result.warnings if validation_result else []
                } if validation_result else None,
                'dry_run': dry_run
            }
            
            self.logger.info(f"Resolution completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Resolution failed: {e}")
            self.stats['failed_resolutions'] += 1
            
            return {
                'success': False,
                'error': str(e),
                'project_path': str(project_path),
                'execution_time': time.time() - start_time
            }
    
    async def resolve_multiple_projects(self, project_paths: List[Path],
                                      dry_run: bool = False,
                                      max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """Resolve dependencies for multiple Flutter projects."""
        self.logger.info(f"Starting batch resolution for {len(project_paths)} projects")
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def resolve_with_semaphore(path):
            async with semaphore:
                return await self.resolve_project(path, dry_run)
        
        # Execute resolutions concurrently
        tasks = [resolve_with_semaphore(path) for path in project_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'success': False,
                    'error': str(result),
                    'project_path': str(project_paths[i])
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _validate_project_structure(self, project_path: Path) -> bool:
        """Validate that the path contains a valid Flutter project."""
        pubspec_path = project_path / 'pubspec.yaml'
        
        if not pubspec_path.exists():
            self.logger.error(f"pubspec.yaml not found in {project_path}")
            return False
        
        # Check for basic Flutter project structure
        required_dirs = ['lib']
        for dir_name in required_dirs:
            if not (project_path / dir_name).exists():
                self.logger.warning(f"Missing {dir_name} directory in {project_path}")
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get resolution statistics."""
        stats = self.stats.copy()
        
        if stats['projects_processed'] > 0:
            stats['success_rate'] = stats['successful_resolutions'] / stats['projects_processed']
            stats['average_execution_time'] = stats['total_execution_time'] / stats['projects_processed']
        else:
            stats['success_rate'] = 0.0
            stats['average_execution_time'] = 0.0
        
        return stats
    
    def save_report(self, results: List[Dict[str, Any]], output_path: Path):
        """Save detailed resolution report."""
        report = {
            'timestamp': time.time(),
            'statistics': self.get_statistics(),
            'results': results,
            'configuration': self.config
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Report saved to {output_path}")


async def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='ML-powered Flutter dependency resolver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resolve dependencies for a single project
  python main.py /path/to/flutter/project
  
  # Dry run (analyze without applying changes)
  python main.py /path/to/flutter/project --dry-run
  
  # Resolve multiple projects
  python main.py /path/to/project1 /path/to/project2 /path/to/project3
  
  # Use custom configuration
  python main.py /path/to/project --config config.json
  
  # Save detailed report
  python main.py /path/to/project --report report.json
        """
    )
    
    parser.add_argument('projects', nargs='+', type=Path,
                       help='Path(s) to Flutter project(s)')
    parser.add_argument('--config', type=Path,
                       help='Path to configuration file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Analyze dependencies without applying changes')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild validation even in dry-run mode')
    parser.add_argument('--max-concurrent', type=int, default=3,
                       help='Maximum concurrent project resolutions')
    parser.add_argument('--report', type=Path,
                       help='Path to save detailed report')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress non-error output')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize resolver
    try:
        resolver = FlutterDependencyResolver(args.config)
    except Exception as e:
        print(f"Failed to initialize resolver: {e}")
        return 1
    
    # Resolve projects
    try:
        if len(args.projects) == 1:
            result = await resolver.resolve_project(
                args.projects[0], 
                dry_run=args.dry_run,
                force_rebuild=args.force_rebuild
            )
            results = [result]
        else:
            results = await resolver.resolve_multiple_projects(
                args.projects,
                dry_run=args.dry_run,
                max_concurrent=args.max_concurrent
            )
        
        # Print summary
        if not args.quiet:
            print("\n" + "="*60)
            print("RESOLUTION SUMMARY")
            print("="*60)
            
            successful = sum(1 for r in results if r['success'])
            failed = len(results) - successful
            
            print(f"Projects processed: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            
            if successful > 0:
                avg_time = sum(r.get('execution_time', 0) for r in results if r['success']) / successful
                print(f"Average execution time: {avg_time:.2f}s")
            
            # Show failed projects
            if failed > 0:
                print(f"\nFailed projects:")
                for result in results:
                    if not result['success']:
                        print(f"  - {result['project_path']}: {result.get('error', 'Unknown error')}")
        
        # Save report if requested
        if args.report:
            resolver.save_report(results, args.report)
        
        # Return appropriate exit code
        return 0 if all(r['success'] for r in results) else 1
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

