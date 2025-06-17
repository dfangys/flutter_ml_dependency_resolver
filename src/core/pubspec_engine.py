"""
Advanced pubspec.yaml parsing and modification engine.
Handles intelligent modification of dependency versions while preserving formatting and comments.
"""

import yaml
import re
import os
import shutil
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass
import logging
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
import semantic_version
from datetime import datetime
import json


@dataclass
class ModificationResult:
    """Result of a pubspec.yaml modification operation."""
    success: bool
    original_content: str
    modified_content: str
    changes_made: List[Dict[str, Any]]
    backup_path: Optional[Path] = None
    error_message: Optional[str] = None


@dataclass
class DependencyChange:
    """Represents a change to a dependency."""
    name: str
    old_version: Optional[str]
    new_version: str
    change_type: str  # 'add', 'update', 'remove'
    section: str  # 'dependencies', 'dev_dependencies', 'dependency_overrides'
    reason: Optional[str] = None


class PubspecModifier:
    """Advanced pubspec.yaml modifier with formatting preservation."""
    
    def __init__(self, preserve_formatting: bool = True, create_backups: bool = True):
        self.preserve_formatting = preserve_formatting
        self.create_backups = create_backups
        self.logger = logging.getLogger(__name__)
        
        # Initialize YAML parser with formatting preservation
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.map_indent = 2
        self.yaml.sequence_indent = 4
        self.yaml.sequence_dash_offset = 2
        self.yaml.width = 4096  # Prevent line wrapping
    
    def modify_pubspec(self, file_path: Path, changes: List[DependencyChange]) -> ModificationResult:
        """Apply dependency changes to a pubspec.yaml file."""
        try:
            # Read original content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Create backup if requested
            backup_path = None
            if self.create_backups:
                backup_path = self._create_backup(file_path, original_content)
            
            # Parse YAML content
            if self.preserve_formatting:
                yaml_data = self.yaml.load(original_content)
            else:
                yaml_data = yaml.safe_load(original_content)
            
            # Apply changes
            changes_made = []
            for change in changes:
                change_result = self._apply_dependency_change(yaml_data, change)
                if change_result:
                    changes_made.append(change_result)
            
            # Generate modified content
            if self.preserve_formatting:
                modified_content = self._dump_yaml_with_formatting(yaml_data)
            else:
                modified_content = yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
            
            # Write modified content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            return ModificationResult(
                success=True,
                original_content=original_content,
                modified_content=modified_content,
                changes_made=changes_made,
                backup_path=backup_path
            )
            
        except Exception as e:
            self.logger.error(f"Failed to modify pubspec.yaml: {e}")
            return ModificationResult(
                success=False,
                original_content="",
                modified_content="",
                changes_made=[],
                error_message=str(e)
            )
    
    def _create_backup(self, file_path: Path, content: str) -> Path:
        """Create a backup of the original file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.parent / f"pubspec.yaml.backup_{timestamp}"
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.logger.info(f"Created backup at {backup_path}")
        return backup_path
    
    def _apply_dependency_change(self, yaml_data: Dict[str, Any], change: DependencyChange) -> Optional[Dict[str, Any]]:
        """Apply a single dependency change to the YAML data."""
        section_name = change.section
        
        # Ensure section exists
        if section_name not in yaml_data:
            if change.change_type == 'add':
                yaml_data[section_name] = CommentedMap() if self.preserve_formatting else {}
            else:
                return None
        
        section = yaml_data[section_name]
        
        if change.change_type == 'add':
            section[change.name] = change.new_version
            return {
                'type': 'add',
                'dependency': change.name,
                'version': change.new_version,
                'section': section_name,
                'reason': change.reason
            }
        
        elif change.change_type == 'update':
            if change.name in section:
                old_value = section[change.name]
                section[change.name] = change.new_version
                return {
                    'type': 'update',
                    'dependency': change.name,
                    'old_version': str(old_value),
                    'new_version': change.new_version,
                    'section': section_name,
                    'reason': change.reason
                }
        
        elif change.change_type == 'remove':
            if change.name in section:
                old_value = section[change.name]
                del section[change.name]
                return {
                    'type': 'remove',
                    'dependency': change.name,
                    'old_version': str(old_value),
                    'section': section_name,
                    'reason': change.reason
                }
        
        return None
    
    def _dump_yaml_with_formatting(self, yaml_data: Dict[str, Any]) -> str:
        """Dump YAML data while preserving formatting."""
        from io import StringIO
        stream = StringIO()
        self.yaml.dump(yaml_data, stream)
        return stream.getvalue()
    
    def validate_pubspec_syntax(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Validate pubspec.yaml syntax."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try parsing with both parsers
            yaml.safe_load(content)
            self.yaml.load(content)
            
            return True, None
            
        except yaml.YAMLError as e:
            return False, f"YAML syntax error: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def restore_from_backup(self, file_path: Path, backup_path: Path) -> bool:
        """Restore pubspec.yaml from backup."""
        try:
            shutil.copy2(backup_path, file_path)
            self.logger.info(f"Restored {file_path} from backup {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restore from backup: {e}")
            return False


class SmartVersionSelector:
    """Intelligent version selection based on constraints and preferences."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def select_optimal_version(self, package_name: str, available_versions: List[str], 
                             constraints: List[str], preferences: Dict[str, Any] = None) -> Optional[str]:
        """Select the optimal version based on constraints and preferences."""
        if not available_versions:
            return None
        
        preferences = preferences or {}
        
        # Filter versions that satisfy all constraints
        compatible_versions = []
        for version in available_versions:
            if self._satisfies_all_constraints(version, constraints):
                compatible_versions.append(version)
        
        if not compatible_versions:
            self.logger.warning(f"No compatible versions found for {package_name}")
            return None
        
        # Sort versions and apply preferences
        sorted_versions = self._sort_versions_by_preference(compatible_versions, preferences)
        
        return sorted_versions[0] if sorted_versions else None
    
    def _satisfies_all_constraints(self, version: str, constraints: List[str]) -> bool:
        """Check if a version satisfies all given constraints."""
        try:
            target_version = semantic_version.Version(version)
            
            for constraint in constraints:
                if not self._satisfies_constraint(target_version, constraint):
                    return False
            
            return True
            
        except ValueError:
            self.logger.warning(f"Invalid version format: {version}")
            return False
    
    def _satisfies_constraint(self, version: semantic_version.Version, constraint: str) -> bool:
        """Check if a version satisfies a single constraint."""
        constraint = constraint.strip()
        
        if constraint == 'any' or constraint == '':
            return True
        
        try:
            # Handle caret syntax
            if constraint.startswith('^'):
                base_version = semantic_version.Version(constraint[1:])
                return (version >= base_version and 
                       version.major == base_version.major)
            
            # Handle range constraints
            if '>=' in constraint:
                min_version = semantic_version.Version(constraint.split('>=')[1].strip())
                return version >= min_version
            elif '<=' in constraint:
                max_version = semantic_version.Version(constraint.split('<=')[1].strip())
                return version <= max_version
            elif '>' in constraint:
                min_version = semantic_version.Version(constraint.split('>')[1].strip())
                return version > min_version
            elif '<' in constraint:
                max_version = semantic_version.Version(constraint.split('<')[1].strip())
                return version < max_version
            else:
                # Exact version
                exact_version = semantic_version.Version(constraint)
                return version == exact_version
                
        except ValueError:
            self.logger.warning(f"Invalid constraint format: {constraint}")
            return False
    
    def _sort_versions_by_preference(self, versions: List[str], preferences: Dict[str, Any]) -> List[str]:
        """Sort versions based on preferences."""
        def version_score(version_str: str) -> Tuple[int, semantic_version.Version]:
            try:
                version = semantic_version.Version(version_str)
                score = 0
                
                # Prefer stable versions
                if not version.prerelease:
                    score += 1000
                
                # Prefer newer versions (but not too new if stability is preferred)
                if preferences.get('prefer_stable', True):
                    score += version.major * 100 + version.minor * 10 + version.patch
                else:
                    score += version.major * 1000 + version.minor * 100 + version.patch * 10
                
                # Avoid alpha/beta versions unless explicitly allowed
                if version.prerelease and not preferences.get('allow_prerelease', False):
                    score -= 10000
                
                return (score, version)
                
            except ValueError:
                return (0, semantic_version.Version('0.0.0'))
        
        return sorted(versions, key=version_score, reverse=True)


class PubspecValidator:
    """Validates pubspec.yaml files for common issues and best practices."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_pubspec(self, file_path: Path) -> Dict[str, Any]:
        """Perform comprehensive validation of pubspec.yaml."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse YAML
            try:
                yaml_data = yaml.safe_load(content)
            except yaml.YAMLError as e:
                validation_result['valid'] = False
                validation_result['errors'].append(f"YAML syntax error: {e}")
                return validation_result
            
            # Validate structure
            self._validate_structure(yaml_data, validation_result)
            
            # Validate dependencies
            self._validate_dependencies(yaml_data, validation_result)
            
            # Validate version constraints
            self._validate_version_constraints(yaml_data, validation_result)
            
            # Check for best practices
            self._check_best_practices(yaml_data, validation_result)
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {e}")
        
        return validation_result
    
    def _validate_structure(self, yaml_data: Dict[str, Any], result: Dict[str, Any]):
        """Validate basic pubspec.yaml structure."""
        required_fields = ['name', 'version']
        
        for field in required_fields:
            if field not in yaml_data:
                result['errors'].append(f"Missing required field: {field}")
                result['valid'] = False
        
        # Check name format
        if 'name' in yaml_data:
            name = yaml_data['name']
            if not re.match(r'^[a-z][a-z0-9_]*$', name):
                result['warnings'].append(
                    f"Package name '{name}' should be lowercase with underscores"
                )
    
    def _validate_dependencies(self, yaml_data: Dict[str, Any], result: Dict[str, Any]):
        """Validate dependency declarations."""
        dependency_sections = ['dependencies', 'dev_dependencies', 'dependency_overrides']
        
        for section in dependency_sections:
            if section in yaml_data:
                deps = yaml_data[section]
                if not isinstance(deps, dict):
                    result['errors'].append(f"{section} must be a map")
                    result['valid'] = False
                    continue
                
                for dep_name, dep_spec in deps.items():
                    self._validate_single_dependency(dep_name, dep_spec, section, result)
    
    def _validate_single_dependency(self, name: str, spec: Any, section: str, result: Dict[str, Any]):
        """Validate a single dependency specification."""
        if isinstance(spec, str):
            # Simple version constraint
            if not self._is_valid_version_constraint(spec):
                result['warnings'].append(
                    f"Invalid version constraint for {name} in {section}: {spec}"
                )
        elif isinstance(spec, dict):
            # Complex dependency specification
            if 'version' in spec and not self._is_valid_version_constraint(spec['version']):
                result['warnings'].append(
                    f"Invalid version constraint for {name} in {section}: {spec['version']}"
                )
        else:
            result['warnings'].append(
                f"Unexpected dependency specification format for {name} in {section}"
            )
    
    def _validate_version_constraints(self, yaml_data: Dict[str, Any], result: Dict[str, Any]):
        """Validate version constraints for potential conflicts."""
        all_deps = {}
        
        # Collect all dependencies
        for section in ['dependencies', 'dev_dependencies']:
            if section in yaml_data:
                for name, spec in yaml_data[section].items():
                    if name not in all_deps:
                        all_deps[name] = []
                    all_deps[name].append((section, spec))
        
        # Check for conflicts
        for dep_name, specs in all_deps.items():
            if len(specs) > 1:
                constraints = []
                for section, spec in specs:
                    if isinstance(spec, str):
                        constraints.append(spec)
                    elif isinstance(spec, dict) and 'version' in spec:
                        constraints.append(spec['version'])
                
                if len(set(constraints)) > 1:
                    result['warnings'].append(
                        f"Conflicting version constraints for {dep_name}: {constraints}"
                    )
    
    def _check_best_practices(self, yaml_data: Dict[str, Any], result: Dict[str, Any]):
        """Check for adherence to best practices."""
        # Check for description
        if 'description' not in yaml_data:
            result['suggestions'].append("Consider adding a description field")
        
        # Check for environment constraints
        if 'environment' not in yaml_data:
            result['suggestions'].append("Consider specifying SDK version constraints in environment section")
        
        # Check for overly broad version constraints
        if 'dependencies' in yaml_data:
            for name, spec in yaml_data['dependencies'].items():
                if isinstance(spec, str) and spec == 'any':
                    result['suggestions'].append(
                        f"Consider using more specific version constraint for {name} instead of 'any'"
                    )
    
    def _is_valid_version_constraint(self, constraint: str) -> bool:
        """Check if a version constraint is valid."""
        if not constraint or constraint == 'any':
            return True
        
        try:
            # Test various constraint formats
            if constraint.startswith('^'):
                semantic_version.Version(constraint[1:])
                return True
            elif any(op in constraint for op in ['>=', '<=', '>', '<']):
                # Range constraint - simplified validation
                return True
            else:
                # Exact version
                semantic_version.Version(constraint)
                return True
        except ValueError:
            return False


class PubspecManager:
    """High-level manager for pubspec.yaml operations."""
    
    def __init__(self, preserve_formatting: bool = True, create_backups: bool = True):
        self.modifier = PubspecModifier(preserve_formatting, create_backups)
        self.validator = PubspecValidator()
        self.version_selector = SmartVersionSelector()
        self.logger = logging.getLogger(__name__)
    
    def apply_ml_resolution(self, file_path: Path, resolution: Dict[str, str], 
                          validation_required: bool = True) -> ModificationResult:
        """Apply ML-generated dependency resolution to pubspec.yaml."""
        # Validate file before modification
        if validation_required:
            is_valid, error = self.modifier.validate_pubspec_syntax(file_path)
            if not is_valid:
                return ModificationResult(
                    success=False,
                    original_content="",
                    modified_content="",
                    changes_made=[],
                    error_message=f"Invalid pubspec.yaml syntax: {error}"
                )
        
        # Convert resolution to dependency changes
        changes = self._resolution_to_changes(file_path, resolution)
        
        # Apply changes
        result = self.modifier.modify_pubspec(file_path, changes)
        
        # Validate result
        if result.success and validation_required:
            is_valid, error = self.modifier.validate_pubspec_syntax(file_path)
            if not is_valid:
                # Restore from backup if validation fails
                if result.backup_path:
                    self.modifier.restore_from_backup(file_path, result.backup_path)
                
                result.success = False
                result.error_message = f"Modified pubspec.yaml is invalid: {error}"
        
        return result
    
    def _resolution_to_changes(self, file_path: Path, resolution: Dict[str, str]) -> List[DependencyChange]:
        """Convert ML resolution to dependency changes."""
        changes = []
        
        try:
            # Read current pubspec.yaml
            with open(file_path, 'r', encoding='utf-8') as f:
                current_data = yaml.safe_load(f)
            
            # Determine changes needed
            for dep_name, new_version in resolution.items():
                change = self._determine_change_type(current_data, dep_name, new_version)
                if change:
                    changes.append(change)
        
        except Exception as e:
            self.logger.error(f"Failed to determine changes: {e}")
        
        return changes
    
    def _determine_change_type(self, current_data: Dict[str, Any], dep_name: str, 
                              new_version: str) -> Optional[DependencyChange]:
        """Determine the type of change needed for a dependency."""
        # Check all dependency sections
        sections = ['dependencies', 'dev_dependencies', 'dependency_overrides']
        
        for section in sections:
            if section in current_data and dep_name in current_data[section]:
                current_spec = current_data[section][dep_name]
                current_version = current_spec if isinstance(current_spec, str) else str(current_spec)
                
                if current_version != new_version:
                    return DependencyChange(
                        name=dep_name,
                        old_version=current_version,
                        new_version=new_version,
                        change_type='update',
                        section=section,
                        reason='ML optimization'
                    )
                return None  # No change needed
        
        # Dependency not found - add to dependencies section
        return DependencyChange(
            name=dep_name,
            old_version=None,
            new_version=new_version,
            change_type='add',
            section='dependencies',
            reason='ML recommendation'
        )
    
    def get_comprehensive_analysis(self, file_path: Path) -> Dict[str, Any]:
        """Get comprehensive analysis of pubspec.yaml file."""
        analysis = {
            'file_path': str(file_path),
            'exists': file_path.exists(),
            'validation': None,
            'dependencies': {},
            'conflicts': [],
            'suggestions': []
        }
        
        if not file_path.exists():
            analysis['error'] = 'File does not exist'
            return analysis
        
        # Validate file
        analysis['validation'] = self.validator.validate_pubspec(file_path)
        
        # Analyze dependencies if file is valid
        if analysis['validation']['valid']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.safe_load(f)
                
                # Extract dependency information
                for section in ['dependencies', 'dev_dependencies', 'dependency_overrides']:
                    if section in yaml_data:
                        analysis['dependencies'][section] = yaml_data[section]
                
                # Additional analysis could be added here
                
            except Exception as e:
                analysis['error'] = f"Analysis failed: {e}"
        
        return analysis


def setup_pubspec_logging():
    """Setup logging for pubspec operations."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


if __name__ == "__main__":
    setup_pubspec_logging()
    
    # Example usage
    manager = PubspecManager()
    
    # This would be used with actual pubspec.yaml files
    # file_path = Path("example_project/pubspec.yaml")
    # analysis = manager.get_comprehensive_analysis(file_path)
    # print(json.dumps(analysis, indent=2))
    
    print("Pubspec.yaml parsing and modification engine initialized successfully")

