"""
Core dependency analysis engine for Flutter projects.
Handles parsing, analysis, and graph construction of pubspec.yaml dependencies.
"""

import yaml
import re
import os
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import networkx as nx
from packaging import version
import semantic_version


@dataclass
class DependencyConstraint:
    """Represents a dependency version constraint."""
    name: str
    constraint: str
    constraint_type: str  # 'caret', 'range', 'exact', 'any'
    min_version: Optional[str] = None
    max_version: Optional[str] = None
    is_dev_dependency: bool = False
    is_override: bool = False
    platform_specific: Optional[str] = None
    
    def __post_init__(self):
        """Parse the constraint string to extract version bounds."""
        self._parse_constraint()
    
    def _parse_constraint(self):
        """Parse version constraint string into structured format."""
        constraint = self.constraint.strip()
        
        if constraint == 'any' or constraint == '':
            self.constraint_type = 'any'
            return
        
        # Handle caret syntax (^1.2.3)
        if constraint.startswith('^'):
            self.constraint_type = 'caret'
            base_version = constraint[1:]
            try:
                parsed = semantic_version.Version(base_version)
                self.min_version = base_version
                # Caret allows patch and minor updates but not major
                self.max_version = f"{parsed.major + 1}.0.0"
            except ValueError:
                logging.warning(f"Invalid caret version: {constraint}")
                self.constraint_type = 'exact'
                self.min_version = self.max_version = base_version
        
        # Handle range syntax (>=1.2.3 <2.0.0)
        elif '>=' in constraint or '<=' in constraint or '>' in constraint or '<' in constraint:
            self.constraint_type = 'range'
            self._parse_range_constraint(constraint)
        
        # Handle exact version
        else:
            self.constraint_type = 'exact'
            self.min_version = self.max_version = constraint
    
    def _parse_range_constraint(self, constraint: str):
        """Parse range constraint like '>=1.2.3 <2.0.0'."""
        # Split on whitespace and parse each part
        parts = constraint.split()
        for part in parts:
            if part.startswith('>='):
                self.min_version = part[2:]
            elif part.startswith('<='):
                self.max_version = part[2:]
            elif part.startswith('>'):
                # Convert > to >= with next patch version
                base_version = part[1:]
                try:
                    parsed = semantic_version.Version(base_version)
                    self.min_version = f"{parsed.major}.{parsed.minor}.{parsed.patch + 1}"
                except ValueError:
                    self.min_version = base_version
            elif part.startswith('<'):
                self.max_version = part[1:]
    
    def satisfies_version(self, version_str: str) -> bool:
        """Check if a given version satisfies this constraint."""
        try:
            target_version = semantic_version.Version(version_str)
            
            if self.constraint_type == 'any':
                return True
            
            if self.constraint_type == 'exact':
                return version_str == self.min_version
            
            # Check minimum version
            if self.min_version:
                min_ver = semantic_version.Version(self.min_version)
                if target_version < min_ver:
                    return False
            
            # Check maximum version
            if self.max_version:
                max_ver = semantic_version.Version(self.max_version)
                if target_version >= max_ver:
                    return False
            
            return True
            
        except ValueError:
            logging.warning(f"Invalid version format: {version_str}")
            return False


@dataclass
class ProjectMetadata:
    """Metadata about a Flutter project."""
    name: str
    version: str
    description: Optional[str] = None
    flutter_sdk_constraint: Optional[str] = None
    dart_sdk_constraint: Optional[str] = None
    platforms: Set[str] = field(default_factory=set)
    project_path: Optional[Path] = None
    
    def __post_init__(self):
        """Initialize default platforms if none specified."""
        if not self.platforms:
            self.platforms = {'android', 'ios'}  # Default Flutter platforms


@dataclass
class DependencyGraph:
    """Represents the complete dependency graph for a project."""
    project_metadata: ProjectMetadata
    dependencies: Dict[str, DependencyConstraint] = field(default_factory=dict)
    dev_dependencies: Dict[str, DependencyConstraint] = field(default_factory=dict)
    dependency_overrides: Dict[str, DependencyConstraint] = field(default_factory=dict)
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    
    def add_dependency(self, dep: DependencyConstraint):
        """Add a dependency to the graph."""
        if dep.is_override:
            self.dependency_overrides[dep.name] = dep
        elif dep.is_dev_dependency:
            self.dev_dependencies[dep.name] = dep
        else:
            self.dependencies[dep.name] = dep
        
        # Add to NetworkX graph
        self.graph.add_node(dep.name, constraint=dep)
    
    def add_dependency_edge(self, parent: str, child: str, constraint_type: str = 'depends'):
        """Add an edge between dependencies."""
        self.graph.add_edge(parent, child, type=constraint_type)
    
    def get_all_dependencies(self) -> Dict[str, DependencyConstraint]:
        """Get all dependencies including overrides."""
        all_deps = {}
        all_deps.update(self.dependencies)
        all_deps.update(self.dev_dependencies)
        all_deps.update(self.dependency_overrides)
        return all_deps
    
    def has_circular_dependencies(self) -> bool:
        """Check if the dependency graph has circular dependencies."""
        try:
            list(nx.topological_sort(self.graph))
            return False
        except nx.NetworkXError:
            return True
    
    def get_dependency_conflicts(self) -> List[Tuple[str, List[DependencyConstraint]]]:
        """Identify potential dependency conflicts."""
        conflicts = []
        all_deps = self.get_all_dependencies()
        
        # Group dependencies by name
        dep_groups = {}
        for dep in all_deps.values():
            if dep.name not in dep_groups:
                dep_groups[dep.name] = []
            dep_groups[dep.name].append(dep)
        
        # Find conflicts
        for name, deps in dep_groups.items():
            if len(deps) > 1:
                # Check if constraints are compatible
                constraints = [dep.constraint for dep in deps]
                if len(set(constraints)) > 1:  # Different constraints
                    conflicts.append((name, deps))
        
        return conflicts


class PubspecParser:
    """Parser for pubspec.yaml files with advanced dependency analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_pubspec_file(self, file_path: Path) -> DependencyGraph:
        """Parse a pubspec.yaml file and return dependency graph."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
            
            return self.parse_pubspec_content(content, file_path.parent)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"pubspec.yaml not found at {file_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in pubspec.yaml: {e}")
    
    def parse_pubspec_content(self, content: Dict[str, Any], project_path: Path) -> DependencyGraph:
        """Parse pubspec.yaml content into dependency graph."""
        # Extract project metadata
        metadata = self._extract_project_metadata(content, project_path)
        
        # Create dependency graph
        dep_graph = DependencyGraph(project_metadata=metadata)
        
        # Parse dependencies
        if 'dependencies' in content:
            self._parse_dependency_section(
                content['dependencies'], dep_graph, is_dev=False
            )
        
        # Parse dev dependencies
        if 'dev_dependencies' in content:
            self._parse_dependency_section(
                content['dev_dependencies'], dep_graph, is_dev=True
            )
        
        # Parse dependency overrides
        if 'dependency_overrides' in content:
            self._parse_dependency_section(
                content['dependency_overrides'], dep_graph, is_override=True
            )
        
        return dep_graph
    
    def _extract_project_metadata(self, content: Dict[str, Any], project_path: Path) -> ProjectMetadata:
        """Extract project metadata from pubspec.yaml content."""
        name = content.get('name', 'unknown')
        version = content.get('version', '0.0.0')
        description = content.get('description')
        
        # Extract SDK constraints
        environment = content.get('environment', {})
        flutter_constraint = environment.get('flutter')
        dart_constraint = environment.get('sdk')
        
        # Detect platforms from project structure
        platforms = self._detect_platforms(project_path)
        
        return ProjectMetadata(
            name=name,
            version=version,
            description=description,
            flutter_sdk_constraint=flutter_constraint,
            dart_sdk_constraint=dart_constraint,
            platforms=platforms,
            project_path=project_path
        )
    
    def _detect_platforms(self, project_path: Path) -> Set[str]:
        """Detect target platforms from project structure."""
        platforms = set()
        
        platform_dirs = {
            'android': 'android',
            'ios': 'ios',
            'web': 'web',
            'windows': 'windows',
            'macos': 'macos',
            'linux': 'linux'
        }
        
        for platform, dir_name in platform_dirs.items():
            if (project_path / dir_name).exists():
                platforms.add(platform)
        
        return platforms if platforms else {'android', 'ios'}  # Default
    
    def _parse_dependency_section(self, deps: Dict[str, Any], dep_graph: DependencyGraph, 
                                 is_dev: bool = False, is_override: bool = False):
        """Parse a dependency section (dependencies, dev_dependencies, etc.)."""
        for name, constraint_info in deps.items():
            if name == 'flutter':
                continue  # Skip Flutter SDK dependency
            
            # Handle different constraint formats
            if isinstance(constraint_info, str):
                constraint = constraint_info
                platform_specific = None
            elif isinstance(constraint_info, dict):
                # Handle complex dependency specifications
                if 'version' in constraint_info:
                    constraint = constraint_info['version']
                elif 'path' in constraint_info:
                    constraint = 'path'  # Local path dependency
                elif 'git' in constraint_info:
                    constraint = 'git'  # Git dependency
                else:
                    constraint = 'any'
                
                platform_specific = constraint_info.get('platform')
            else:
                constraint = 'any'
                platform_specific = None
            
            # Create dependency constraint
            dep_constraint = DependencyConstraint(
                name=name,
                constraint=constraint,
                constraint_type='',  # Will be set in __post_init__
                is_dev_dependency=is_dev,
                is_override=is_override,
                platform_specific=platform_specific
            )
            
            dep_graph.add_dependency(dep_constraint)


class DependencyAnalyzer:
    """Advanced dependency analysis and conflict detection."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.parser = PubspecParser()
    
    def analyze_project(self, project_path: Path) -> DependencyGraph:
        """Analyze a single Flutter project."""
        pubspec_path = project_path / 'pubspec.yaml'
        return self.parser.parse_pubspec_file(pubspec_path)
    
    def analyze_multiple_projects(self, project_paths: List[Path]) -> List[DependencyGraph]:
        """Analyze multiple Flutter projects."""
        graphs = []
        for path in project_paths:
            try:
                graph = self.analyze_project(path)
                graphs.append(graph)
            except Exception as e:
                self.logger.error(f"Failed to analyze project at {path}: {e}")
        
        return graphs
    
    def find_cross_project_conflicts(self, graphs: List[DependencyGraph]) -> Dict[str, List[DependencyConstraint]]:
        """Find dependency conflicts across multiple projects."""
        all_dependencies = {}
        
        # Collect all dependencies from all projects
        for graph in graphs:
            for dep_name, dep_constraint in graph.get_all_dependencies().items():
                if dep_name not in all_dependencies:
                    all_dependencies[dep_name] = []
                all_dependencies[dep_name].append(dep_constraint)
        
        # Find conflicts
        conflicts = {}
        for dep_name, constraints in all_dependencies.items():
            if len(constraints) > 1:
                # Check if all constraints are compatible
                unique_constraints = set(c.constraint for c in constraints)
                if len(unique_constraints) > 1:
                    conflicts[dep_name] = constraints
        
        return conflicts
    
    def suggest_version_resolution(self, conflicts: Dict[str, List[DependencyConstraint]]) -> Dict[str, str]:
        """Suggest version resolutions for conflicts."""
        suggestions = {}
        
        for dep_name, constraints in conflicts.items():
            # Find the most restrictive constraint that satisfies all requirements
            suggested_version = self._find_compatible_version(constraints)
            if suggested_version:
                suggestions[dep_name] = suggested_version
            else:
                # If no compatible version found, suggest the latest constraint
                latest_constraint = max(constraints, key=lambda c: c.constraint)
                suggestions[dep_name] = latest_constraint.constraint
        
        return suggestions
    
    def _find_compatible_version(self, constraints: List[DependencyConstraint]) -> Optional[str]:
        """Find a version that satisfies all constraints."""
        # This is a simplified implementation
        # In a real system, this would query package repositories for available versions
        
        # For now, return the most restrictive constraint
        exact_constraints = [c for c in constraints if c.constraint_type == 'exact']
        if exact_constraints:
            # If there are exact constraints, they must all be the same
            versions = set(c.constraint for c in exact_constraints)
            if len(versions) == 1:
                return list(versions)[0]
            else:
                return None  # Conflicting exact versions
        
        # Handle caret and range constraints
        # This would need more sophisticated logic in a real implementation
        return None


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


if __name__ == "__main__":
    setup_logging()
    
    # Example usage
    analyzer = DependencyAnalyzer()
    
    # This would be used to analyze actual Flutter projects
    # project_path = Path("/path/to/flutter/project")
    # graph = analyzer.analyze_project(project_path)
    # print(f"Analyzed project: {graph.project_metadata.name}")
    # print(f"Dependencies: {len(graph.dependencies)}")
    # print(f"Conflicts: {graph.get_dependency_conflicts()}")

