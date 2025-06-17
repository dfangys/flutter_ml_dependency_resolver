"""
ML-powered version resolution algorithms.
Integrates dependency analysis, machine learning, and constraint solving for optimal dependency resolution.
"""

import asyncio
import aiohttp
import json
import time
import random
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
from packaging import version
import semantic_version
from itertools import combinations, product
import math

# Import our custom modules
import sys
sys.path.append('/home/ubuntu/flutter_ml_dependency_resolver/src')

from analysis.dependency_analyzer import DependencyAnalyzer, DependencyGraph, DependencyConstraint
from ml.ml_core import DependencyResolutionAgent, MLConfig, EnsembleResolver
from core.pubspec_engine import PubspecManager, DependencyChange


@dataclass
class PackageInfo:
    """Information about a package from pub.dev."""
    name: str
    versions: List[str]
    latest_version: str
    description: Optional[str] = None
    popularity_score: float = 0.0
    pub_points: int = 0
    likes: int = 0
    dependencies: Dict[str, str] = field(default_factory=dict)
    dev_dependencies: Dict[str, str] = field(default_factory=dict)
    platforms: Set[str] = field(default_factory=set)
    sdk_constraints: Dict[str, str] = field(default_factory=dict)


@dataclass
class ResolutionCandidate:
    """A candidate dependency resolution."""
    resolution: Dict[str, str]
    score: float
    confidence: float
    conflicts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolutionResult:
    """Result of dependency resolution process."""
    success: bool
    best_candidate: Optional[ResolutionCandidate]
    all_candidates: List[ResolutionCandidate]
    execution_time: float
    iterations: int
    error_message: Optional[str] = None
    package_info: Dict[str, PackageInfo] = field(default_factory=dict)


class PackageRepository:
    """Interface to package repositories (pub.dev, etc.)."""
    
    def __init__(self, cache_ttl: int = 3600):
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.logger = logging.getLogger(__name__)
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def get_package_info(self, package_name: str) -> Optional[PackageInfo]:
        """Get package information from pub.dev."""
        cache_key = f"package_info_{package_name}"
        
        # Check cache
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_data
        
        try:
            # Fetch from pub.dev API
            url = f"https://pub.dev/api/packages/{package_name}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    package_info = self._parse_package_data(data)
                    
                    # Cache result
                    self.cache[cache_key] = (time.time(), package_info)
                    return package_info
                else:
                    self.logger.warning(f"Failed to fetch package info for {package_name}: {response.status}")
                    return None
        
        except Exception as e:
            self.logger.error(f"Error fetching package info for {package_name}: {e}")
            return None
    
    def _parse_package_data(self, data: Dict[str, Any]) -> PackageInfo:
        """Parse package data from pub.dev API response."""
        latest = data.get('latest', {})
        pubspec = latest.get('pubspec', {})
        
        # Extract versions
        versions = [v['version'] for v in data.get('versions', [])]
        versions.sort(key=lambda v: semantic_version.Version(v), reverse=True)
        
        # Extract dependencies
        dependencies = pubspec.get('dependencies', {})
        dev_dependencies = pubspec.get('dev_dependencies', {})
        
        # Extract platform support
        platforms = set()
        if 'flutter' in dependencies:
            platforms.update(['android', 'ios'])
        if 'dart:html' in str(pubspec):
            platforms.add('web')
        
        # Extract SDK constraints
        environment = pubspec.get('environment', {})
        sdk_constraints = {
            'dart': environment.get('sdk', ''),
            'flutter': environment.get('flutter', '')
        }
        
        return PackageInfo(
            name=data['name'],
            versions=versions,
            latest_version=latest.get('version', ''),
            description=pubspec.get('description', ''),
            dependencies={k: str(v) for k, v in dependencies.items()},
            dev_dependencies={k: str(v) for k, v in dev_dependencies.items()},
            platforms=platforms,
            sdk_constraints=sdk_constraints
        )
    
    async def get_multiple_package_info(self, package_names: List[str]) -> Dict[str, PackageInfo]:
        """Get information for multiple packages concurrently."""
        tasks = [self.get_package_info(name) for name in package_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        package_info = {}
        for name, result in zip(package_names, results):
            if isinstance(result, PackageInfo):
                package_info[name] = result
            elif isinstance(result, Exception):
                self.logger.error(f"Failed to get info for {name}: {result}")
        
        return package_info


class ConstraintSolver:
    """Advanced constraint solver for dependency resolution."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def solve_constraints(self, dependency_graph: DependencyGraph, 
                         package_info: Dict[str, PackageInfo]) -> List[Dict[str, str]]:
        """Solve dependency constraints to find valid resolutions."""
        all_deps = dependency_graph.get_all_dependencies()
        
        # Build constraint satisfaction problem
        variables = {}  # package_name -> list of valid versions
        constraints = []  # list of constraint functions
        
        for dep_name, dep_constraint in all_deps.items():
            if dep_name in package_info:
                pkg_info = package_info[dep_name]
                valid_versions = self._filter_valid_versions(dep_constraint, pkg_info.versions)
                variables[dep_name] = valid_versions
            else:
                # Package not found - use constraint as-is
                variables[dep_name] = [dep_constraint.constraint]
        
        # Add transitive dependency constraints
        self._add_transitive_constraints(variables, package_info, constraints)
        
        # Generate candidate solutions
        candidates = self._generate_candidates(variables, constraints, max_candidates=100)
        
        return candidates
    
    def _filter_valid_versions(self, constraint: DependencyConstraint, 
                              available_versions: List[str]) -> List[str]:
        """Filter versions that satisfy the constraint."""
        valid_versions = []
        
        for version_str in available_versions:
            if constraint.satisfies_version(version_str):
                valid_versions.append(version_str)
        
        return valid_versions
    
    def _add_transitive_constraints(self, variables: Dict[str, List[str]], 
                                   package_info: Dict[str, PackageInfo], 
                                   constraints: List[Any]):
        """Add constraints from transitive dependencies."""
        # This is a simplified implementation
        # In practice, this would recursively resolve all transitive dependencies
        
        for pkg_name, pkg_info in package_info.items():
            if pkg_name in variables:
                # Add constraints from this package's dependencies
                for dep_name, dep_constraint in pkg_info.dependencies.items():
                    if dep_name in variables:
                        # Create constraint function
                        def constraint_func(assignment, pkg=pkg_name, dep=dep_name, constraint=dep_constraint):
                            if pkg in assignment and dep in assignment:
                                # Check if the selected version of pkg is compatible with dep constraint
                                return self._check_compatibility(assignment[pkg], dep, constraint)
                            return True
                        
                        constraints.append(constraint_func)
    
    def _check_compatibility(self, pkg_version: str, dep_name: str, dep_constraint: str) -> bool:
        """Check if a package version is compatible with a dependency constraint."""
        # Simplified compatibility check
        # In practice, this would parse the constraint and check compatibility
        return True  # Placeholder
    
    def _generate_candidates(self, variables: Dict[str, List[str]], 
                           constraints: List[Any], max_candidates: int = 100) -> List[Dict[str, str]]:
        """Generate candidate solutions using constraint satisfaction."""
        candidates = []
        
        # Get all variable names
        var_names = list(variables.keys())
        
        if not var_names:
            return candidates
        
        # Use iterative approach for large search spaces
        max_combinations = min(max_candidates * 10, 10000)
        
        # Generate combinations using sampling for large spaces
        total_combinations = 1
        for versions in variables.values():
            total_combinations *= len(versions)
        
        if total_combinations <= max_combinations:
            # Small search space - enumerate all combinations
            for combination in product(*[variables[name] for name in var_names]):
                assignment = dict(zip(var_names, combination))
                if self._satisfies_constraints(assignment, constraints):
                    candidates.append(assignment)
                    if len(candidates) >= max_candidates:
                        break
        else:
            # Large search space - use random sampling
            for _ in range(max_combinations):
                assignment = {}
                for name in var_names:
                    assignment[name] = random.choice(variables[name])
                
                if self._satisfies_constraints(assignment, constraints):
                    candidates.append(assignment)
                    if len(candidates) >= max_candidates:
                        break
        
        return candidates
    
    def _satisfies_constraints(self, assignment: Dict[str, str], constraints: List[Any]) -> bool:
        """Check if an assignment satisfies all constraints."""
        for constraint_func in constraints:
            try:
                if not constraint_func(assignment):
                    return False
            except Exception:
                # If constraint evaluation fails, assume it's not satisfied
                return False
        
        return True


class MLVersionResolver:
    """Main ML-powered version resolver."""
    
    def __init__(self, config: MLConfig = None):
        self.config = config or MLConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.dependency_analyzer = DependencyAnalyzer()
        self.ml_agent = DependencyResolutionAgent(self.config)
        self.ensemble_resolver = EnsembleResolver(self.config)
        self.pubspec_manager = PubspecManager()
        self.constraint_solver = ConstraintSolver()
        
        # Training data and performance tracking
        self.training_data = []
        self.resolution_history = []
        self.performance_metrics = {
            'total_resolutions': 0,
            'successful_resolutions': 0,
            'average_resolution_time': 0.0,
            'average_score': 0.0
        }
    
    async def resolve_dependencies(self, project_path: Path, 
                                 optimization_goals: Dict[str, float] = None) -> ResolutionResult:
        """Resolve dependencies for a Flutter project using ML."""
        start_time = time.time()
        
        try:
            # Analyze project dependencies
            dependency_graph = self.dependency_analyzer.analyze_project(project_path)
            
            # Get package information
            all_deps = dependency_graph.get_all_dependencies()
            package_names = list(all_deps.keys())
            
            async with PackageRepository() as repo:
                package_info = await repo.get_multiple_package_info(package_names)
            
            # Generate candidate resolutions
            candidates = await self._generate_resolution_candidates(
                dependency_graph, package_info, optimization_goals
            )
            
            # Evaluate and rank candidates
            ranked_candidates = self._rank_candidates(candidates, dependency_graph, optimization_goals)
            
            # Select best candidate
            best_candidate = ranked_candidates[0] if ranked_candidates else None
            
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self._update_performance_metrics(execution_time, best_candidate)
            
            return ResolutionResult(
                success=best_candidate is not None,
                best_candidate=best_candidate,
                all_candidates=ranked_candidates,
                execution_time=execution_time,
                iterations=len(candidates),
                package_info=package_info
            )
        
        except Exception as e:
            self.logger.error(f"Resolution failed: {e}")
            return ResolutionResult(
                success=False,
                best_candidate=None,
                all_candidates=[],
                execution_time=time.time() - start_time,
                iterations=0,
                error_message=str(e)
            )
    
    async def _generate_resolution_candidates(self, dependency_graph: DependencyGraph,
                                            package_info: Dict[str, PackageInfo],
                                            optimization_goals: Dict[str, float] = None) -> List[ResolutionCandidate]:
        """Generate multiple resolution candidates using different strategies."""
        candidates = []
        
        # Strategy 1: Constraint solver
        constraint_solutions = self.constraint_solver.solve_constraints(dependency_graph, package_info)
        for solution in constraint_solutions[:20]:  # Limit to top 20
            candidate = ResolutionCandidate(
                resolution=solution,
                score=0.0,  # Will be calculated later
                confidence=0.8,
                metadata={'strategy': 'constraint_solver'}
            )
            candidates.append(candidate)
        
        # Strategy 2: ML agent
        ml_solutions = await self._generate_ml_solutions(dependency_graph, package_info, count=10)
        candidates.extend(ml_solutions)
        
        # Strategy 3: Ensemble approach
        ensemble_solutions = await self._generate_ensemble_solutions(dependency_graph, package_info, count=5)
        candidates.extend(ensemble_solutions)
        
        # Strategy 4: Conservative approach (latest stable versions)
        conservative_solution = self._generate_conservative_solution(dependency_graph, package_info)
        if conservative_solution:
            candidates.append(conservative_solution)
        
        # Strategy 5: Aggressive approach (latest versions including prereleases)
        aggressive_solution = self._generate_aggressive_solution(dependency_graph, package_info)
        if aggressive_solution:
            candidates.append(aggressive_solution)
        
        return candidates
    
    async def _generate_ml_solutions(self, dependency_graph: DependencyGraph,
                                   package_info: Dict[str, PackageInfo],
                                   count: int = 10) -> List[ResolutionCandidate]:
        """Generate solutions using ML agent."""
        solutions = []
        
        for i in range(count):
            resolution = {}
            all_deps = dependency_graph.get_all_dependencies()
            
            for dep_name in all_deps.keys():
                if dep_name in package_info:
                    available_versions = package_info[dep_name].versions
                    selected_version = self.ml_agent.select_dependency_version(
                        dependency_graph, dep_name, available_versions
                    )
                    resolution[dep_name] = selected_version
            
            if resolution:
                candidate = ResolutionCandidate(
                    resolution=resolution,
                    score=0.0,
                    confidence=0.7,
                    metadata={'strategy': 'ml_agent', 'iteration': i}
                )
                solutions.append(candidate)
        
        return solutions
    
    async def _generate_ensemble_solutions(self, dependency_graph: DependencyGraph,
                                         package_info: Dict[str, PackageInfo],
                                         count: int = 5) -> List[ResolutionCandidate]:
        """Generate solutions using ensemble approach."""
        solutions = []
        
        # Prepare available versions for ensemble
        available_versions = {}
        for dep_name, pkg_info in package_info.items():
            available_versions[dep_name] = pkg_info.versions
        
        for i in range(count):
            resolution = self.ensemble_resolver.resolve_dependencies(dependency_graph, available_versions)
            
            if resolution:
                candidate = ResolutionCandidate(
                    resolution=resolution,
                    score=0.0,
                    confidence=0.9,
                    metadata={'strategy': 'ensemble', 'iteration': i}
                )
                solutions.append(candidate)
        
        return solutions
    
    def _generate_conservative_solution(self, dependency_graph: DependencyGraph,
                                      package_info: Dict[str, PackageInfo]) -> Optional[ResolutionCandidate]:
        """Generate conservative solution using latest stable versions."""
        resolution = {}
        all_deps = dependency_graph.get_all_dependencies()
        
        for dep_name, dep_constraint in all_deps.items():
            if dep_name in package_info:
                pkg_info = package_info[dep_name]
                
                # Find latest stable version that satisfies constraint
                for version_str in pkg_info.versions:
                    try:
                        version_obj = semantic_version.Version(version_str)
                        if not version_obj.prerelease and dep_constraint.satisfies_version(version_str):
                            resolution[dep_name] = version_str
                            break
                    except ValueError:
                        continue
                
                # Fallback to any compatible version
                if dep_name not in resolution:
                    for version_str in pkg_info.versions:
                        if dep_constraint.satisfies_version(version_str):
                            resolution[dep_name] = version_str
                            break
        
        if resolution:
            return ResolutionCandidate(
                resolution=resolution,
                score=0.0,
                confidence=0.6,
                metadata={'strategy': 'conservative'}
            )
        
        return None
    
    def _generate_aggressive_solution(self, dependency_graph: DependencyGraph,
                                    package_info: Dict[str, PackageInfo]) -> Optional[ResolutionCandidate]:
        """Generate aggressive solution using latest versions including prereleases."""
        resolution = {}
        all_deps = dependency_graph.get_all_dependencies()
        
        for dep_name, dep_constraint in all_deps.items():
            if dep_name in package_info:
                pkg_info = package_info[dep_name]
                
                # Find latest version that satisfies constraint
                for version_str in pkg_info.versions:
                    if dep_constraint.satisfies_version(version_str):
                        resolution[dep_name] = version_str
                        break
        
        if resolution:
            return ResolutionCandidate(
                resolution=resolution,
                score=0.0,
                confidence=0.5,
                metadata={'strategy': 'aggressive'}
            )
        
        return None
    
    def _rank_candidates(self, candidates: List[ResolutionCandidate],
                        dependency_graph: DependencyGraph,
                        optimization_goals: Dict[str, float] = None) -> List[ResolutionCandidate]:
        """Rank resolution candidates by quality score."""
        optimization_goals = optimization_goals or {
            'stability': 0.4,
            'performance': 0.3,
            'security': 0.2,
            'compatibility': 0.1
        }
        
        # Calculate scores for all candidates
        for candidate in candidates:
            candidate.score = self._calculate_candidate_score(
                candidate, dependency_graph, optimization_goals
            )
        
        # Sort by score (descending)
        ranked_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        
        return ranked_candidates
    
    def _calculate_candidate_score(self, candidate: ResolutionCandidate,
                                  dependency_graph: DependencyGraph,
                                  optimization_goals: Dict[str, float]) -> float:
        """Calculate quality score for a resolution candidate."""
        score = 0.0
        
        # Base score from ML agent evaluation
        base_score = self.ml_agent.evaluate_resolution(dependency_graph, candidate.resolution)
        score += base_score * 0.5
        
        # Stability score
        stability_score = self._calculate_stability_score(candidate.resolution)
        score += stability_score * optimization_goals.get('stability', 0.4)
        
        # Compatibility score
        compatibility_score = self._calculate_compatibility_score(candidate.resolution, dependency_graph)
        score += compatibility_score * optimization_goals.get('compatibility', 0.3)
        
        # Security score (simplified)
        security_score = self._calculate_security_score(candidate.resolution)
        score += security_score * optimization_goals.get('security', 0.2)
        
        # Performance score (simplified)
        performance_score = self._calculate_performance_score(candidate.resolution)
        score += performance_score * optimization_goals.get('performance', 0.1)
        
        return score
    
    def _calculate_stability_score(self, resolution: Dict[str, str]) -> float:
        """Calculate stability score based on version choices."""
        if not resolution:
            return 0.0
        
        stable_count = 0
        for version_str in resolution.values():
            try:
                version_obj = semantic_version.Version(version_str)
                if not version_obj.prerelease:
                    stable_count += 1
            except ValueError:
                # If version parsing fails, assume it's unstable
                pass
        
        return stable_count / len(resolution)
    
    def _calculate_compatibility_score(self, resolution: Dict[str, str],
                                     dependency_graph: DependencyGraph) -> float:
        """Calculate compatibility score."""
        all_deps = dependency_graph.get_all_dependencies()
        compatible_count = 0
        
        for dep_name, selected_version in resolution.items():
            if dep_name in all_deps:
                if all_deps[dep_name].satisfies_version(selected_version):
                    compatible_count += 1
        
        return compatible_count / len(resolution) if resolution else 0.0
    
    def _calculate_security_score(self, resolution: Dict[str, str]) -> float:
        """Calculate security score (simplified implementation)."""
        # In a real implementation, this would check against vulnerability databases
        # For now, prefer newer versions as they're more likely to have security fixes
        
        if not resolution:
            return 0.0
        
        recent_count = 0
        for version_str in resolution.values():
            try:
                version_obj = semantic_version.Version(version_str)
                # Simple heuristic: versions with higher major/minor numbers are more recent
                if version_obj.major >= 1 and version_obj.minor >= 0:
                    recent_count += 1
            except ValueError:
                pass
        
        return recent_count / len(resolution)
    
    def _calculate_performance_score(self, resolution: Dict[str, str]) -> float:
        """Calculate performance score (simplified implementation)."""
        # In a real implementation, this would consider package size, build time, etc.
        # For now, use a simple heuristic
        
        if not resolution:
            return 0.0
        
        # Prefer stable versions as they're typically more optimized
        return self._calculate_stability_score(resolution)
    
    def _update_performance_metrics(self, execution_time: float, best_candidate: Optional[ResolutionCandidate]):
        """Update performance tracking metrics."""
        self.performance_metrics['total_resolutions'] += 1
        
        if best_candidate:
            self.performance_metrics['successful_resolutions'] += 1
            
            # Update average score
            current_avg_score = self.performance_metrics['average_score']
            total_successful = self.performance_metrics['successful_resolutions']
            new_avg_score = ((current_avg_score * (total_successful - 1)) + best_candidate.score) / total_successful
            self.performance_metrics['average_score'] = new_avg_score
        
        # Update average resolution time
        current_avg_time = self.performance_metrics['average_resolution_time']
        total_resolutions = self.performance_metrics['total_resolutions']
        new_avg_time = ((current_avg_time * (total_resolutions - 1)) + execution_time) / total_resolutions
        self.performance_metrics['average_resolution_time'] = new_avg_time
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report."""
        success_rate = 0.0
        if self.performance_metrics['total_resolutions'] > 0:
            success_rate = (self.performance_metrics['successful_resolutions'] / 
                          self.performance_metrics['total_resolutions'])
        
        return {
            'total_resolutions': self.performance_metrics['total_resolutions'],
            'successful_resolutions': self.performance_metrics['successful_resolutions'],
            'success_rate': success_rate,
            'average_resolution_time': self.performance_metrics['average_resolution_time'],
            'average_score': self.performance_metrics['average_score']
        }
    
    async def apply_resolution(self, project_path: Path, resolution_result: ResolutionResult) -> bool:
        """Apply the best resolution to the project's pubspec.yaml."""
        if not resolution_result.success or not resolution_result.best_candidate:
            return False
        
        try:
            pubspec_path = project_path / 'pubspec.yaml'
            modification_result = self.pubspec_manager.apply_ml_resolution(
                pubspec_path, resolution_result.best_candidate.resolution
            )
            
            return modification_result.success
        
        except Exception as e:
            self.logger.error(f"Failed to apply resolution: {e}")
            return False


def setup_resolver_logging():
    """Setup logging for resolver components."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


if __name__ == "__main__":
    setup_resolver_logging()
    
    # Example usage
    resolver = MLVersionResolver()
    
    print("ML-powered version resolver initialized successfully")
    print(f"Configuration: {resolver.config}")
    print("Ready to resolve dependencies!")

