"""
Advanced dependency analysis for Flutter projects.
Handles complex dependency graphs, version constraints, and conflict detection.
"""

import yaml
import logging
import networkx as nx
import semantic_version
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DependencyConstraint:
    """Represents a dependency version constraint."""
    name: str
    constraint: str
    constraint_type: str = field(init=False)
    min_version: Optional[str] = field(default=None, init=False)
    max_version: Optional[str] = field(default=None, init=False)
    is_dev_dependency: bool = False
    is_override: bool = False
    platform_specific: Optional[str] = None
    
    def __post_init__(self):
        """Parse constraint after initialization."""
        # Handle special constraint types
        if self.constraint in ['any', 'path', 'git', 'sdk']:
            self.constraint_type = self.constraint
            return
        
        # Ensure constraint is a string
        if not isinstance(self.constraint, str):
            logging.warning(f"Non-string constraint for {self.name}: {self.constraint}")
            self.constraint = str(self.constraint)
        
        # Handle caret syntax (^1.2.3)
        if self.constraint.startswith('^'):
            self.constraint_type = 'caret'
            base_version = self.constraint[1:]
            try:
                parsed = semantic_version.Version(base_version)
                self.min_version = base_version
                self.max_version = f"{parsed.major + 1}.0.0"
            except ValueError:
                logging.warning(f"Invalid caret version for {self.name}: {self.constraint}")
                self.constraint_type = 'exact'
                self.min_version = self.max_version = base_version
        
        # Handle range syntax (>=1.2.3 <2.0.0)
        elif '>=' in self.constraint or '<=' in self.constraint or '>' in self.constraint or '<' in self.constraint:
            self.constraint_type = 'range'
            self._parse_range_constraint(self.constraint)
        
        # Handle exact version
        else:
            self.constraint_type = 'exact'
            self.min_version = self.max_version = self.constraint
    
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
            
            if self.constraint_type in ['path', 'git', 'sdk']:
                return True  # These are always satisfied for resolution purposes
            
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
            # If version parsing fails, assume it doesn't satisfy
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


class DependencyGraph:
    """Represents a dependency graph for a Flutter project."""
    
    def __init__(self, project_metadata: ProjectMetadata):
        self.project_metadata = project_metadata
        self.dependencies: Dict[str, DependencyConstraint] = {}
        self.dev_dependencies: Dict[str, DependencyConstraint] = {}
        self.dependency_overrides: Dict[str, DependencyConstraint] = {}
        
        # NetworkX graph for advanced analysis
        self.graph = nx.DiGraph()
    
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
    
    def parse_pubspec_file(self, pubspec_path: Path) -> DependencyGraph:
        """Parse a pubspec.yaml file and return dependency graph."""
        try:
            with open(pubspec_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
            
            return self.parse_pubspec_content(content, pubspec_path.parent)
            
        except Exception as e:
            self.logger.error(f"Failed to parse {pubspec_path}: {e}")
            raise
    
    def parse_pubspec_content(self, content: Dict[str, Any], project_path: Path) -> DependencyGraph:
        """Parse pubspec content and return dependency graph."""
        # Extract project metadata
        metadata = self._extract_project_metadata(content, project_path)
        
        # Create dependency graph
        dep_graph = DependencyGraph(metadata)
        
        # Parse dependency sections
        if 'dependencies' in content:
            self._parse_dependency_section(content['dependencies'], dep_graph, is_dev=False)
        
        if 'dev_dependencies' in content:
            self._parse_dependency_section(content['dev_dependencies'], dep_graph, is_dev=True)
        
        if 'dependency_overrides' in content:
            self._parse_dependency_section(content['dependency_overrides'], dep_graph, is_override=True)
        
        return dep_graph
    
    def _extract_project_metadata(self, content: Dict[str, Any], project_path: Path) -> ProjectMetadata:
        """Extract project metadata from pubspec content."""
        name = content.get('name', 'unknown')
        version = content.get('version', '0.0.0')
        description = content.get('description')
        
        # Extract environment constraints
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
                    # Ensure we extract the version string properly
                    version_value = constraint_info['version']
                    if isinstance(version_value, str):
                        constraint = version_value
                    else:
                        # Handle nested version objects
                        constraint = str(version_value)
                elif 'path' in constraint_info:
                    constraint = 'path'  # Local path dependency
                elif 'git' in constraint_info:
                    constraint = 'git'  # Git dependency
                elif 'sdk' in constraint_info:
                    constraint = 'sdk'  # SDK dependency
                else:
                    constraint = 'any'
                
                platform_specific = constraint_info.get('platform')
            else:
                # Handle other types by converting to string
                constraint = str(constraint_info) if constraint_info is not None else 'any'
                platform_specific = None
            
            # Create dependency constraint
            dep_constraint = DependencyConstraint(
                name=name,
                constraint=constraint,
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
    
    def find_cross_project_conflicts(self, graphs: List[DependencyGraph]) -> List[Dict[str, Any]]:
        """Find conflicts across multiple projects."""
        conflicts = []
        
        # Collect all dependencies across projects
        all_project_deps = {}
        for graph in graphs:
            project_name = graph.project_metadata.name
            all_project_deps[project_name] = graph.get_all_dependencies()
        
        # Find packages used in multiple projects with different constraints
        package_usage = {}
        for project_name, deps in all_project_deps.items():
            for dep_name, dep_constraint in deps.items():
                if dep_name not in package_usage:
                    package_usage[dep_name] = []
                package_usage[dep_name].append((project_name, dep_constraint))
        
        # Identify conflicts
        for package_name, usage_list in package_usage.items():
            if len(usage_list) > 1:
                constraints = [usage[1].constraint for usage in usage_list]
                if len(set(constraints)) > 1:
                    conflicts.append({
                        'package': package_name,
                        'usage': usage_list,
                        'conflict_type': 'cross_project_version_mismatch'
                    })
        
        return conflicts
    
    def suggest_resolution_strategy(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Suggest resolution strategies for conflicts."""
        strategies = {
            'unified_versions': {},
            'dependency_overrides': {},
            'platform_specific': {}
        }
        
        for conflict in conflicts:
            package_name = conflict['package']
            usage_list = conflict['usage']
            
            # Find the most restrictive compatible version
            all_constraints = [usage[1] for usage in usage_list]
            
            # Simple strategy: use the highest version that satisfies all constraints
            # This is a simplified implementation - a real resolver would be more sophisticated
            versions = []
            for constraint in all_constraints:
                if constraint.constraint_type == 'exact':
                    versions.append(constraint.min_version)
                elif constraint.constraint_type == 'caret' and constraint.min_version:
                    versions.append(constraint.min_version)
            
            if versions:
                # Sort versions and pick the highest
                try:
                    sorted_versions = sorted(versions, key=semantic_version.Version, reverse=True)
                    strategies['unified_versions'][package_name] = sorted_versions[0]
                except ValueError:
                    # If version parsing fails, use the first version
                    strategies['unified_versions'][package_name] = versions[0]
        
        return strategies


def setup_dependency_logging():
    """Setup logging for dependency analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


if __name__ == "__main__":
    setup_dependency_logging()
    
    # Example usage
    analyzer = DependencyAnalyzer()
    
    # This would be used with actual project paths
    # project_path = Path("/path/to/flutter/project")
    # graph = analyzer.analyze_project(project_path)
    # print(f"Analyzed project: {graph.project_metadata.name}")
    # print(f"Dependencies: {len(graph.dependencies)}")
    # print(f"Conflicts: {len(graph.get_dependency_conflicts())}")
    
    print("Dependency analyzer initialized successfully")

