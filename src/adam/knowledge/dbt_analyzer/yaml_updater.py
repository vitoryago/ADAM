"""
YAML Updater for DBT Projects
Handles safe in-place updates of YAML files while preserving comments and formatting.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import PreservedScalarString

logger = logging.getLogger(__name__)

class YAMLUpdater:
    """
    Handles safe updates to YAML files using ruamel.yaml to preserve
    comments and formatting.
    """
    
    def __init__(self):
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.indent(mapping=2, sequence=4, offset=2)
        self.yaml.width = 4096  # Prevent line wrapping for long descriptions
    
    def update_column_description(self, 
                                file_path: Path, 
                                model_name: str, 
                                column_name: str, 
                                description: str) -> bool:
        """
        Update a column description in a specific file.
        Returns True if successful, False otherwise.
        """
        try:
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
                
            # Load YAML
            with open(file_path, 'r') as f:
                data = self.yaml.load(f)
            
            if not data or 'models' not in data:
                return False
                
            # Find model
            model_node = None
            for model in data['models']:
                if model.get('name') == model_name:
                    model_node = model
                    break
            
            if not model_node:
                return False
                
            # Ensure columns list exists
            if 'columns' not in model_node:
                model_node['columns'] = []
                
            # Find or create column
            column_node = None
            for col in model_node['columns']:
                if col.get('name') == column_name:
                    column_node = col
                    break
            
            if column_node:
                # Update existing
                if description and '\n' in description:
                    column_node['description'] = PreservedScalarString(description)
                else:
                    column_node['description'] = description
            else:
                # Create new
                new_col = {'name': column_name}
                if description and '\n' in description:
                    new_col['description'] = PreservedScalarString(description)
                else:
                    new_col['description'] = description
                model_node['columns'].append(new_col)
            
            # Save back to file
            with open(file_path, 'w') as f:
                self.yaml.dump(data, f)
                
            return True
            
        except Exception as e:
            logger.error(f"Error updating YAML {file_path}: {e}")
            return False

    def batch_update_descriptions(self, 
                                file_path: Path, 
                                updates: Dict[str, Dict[str, str]]) -> int:
        """
        Update multiple descriptions in a single file.
        updates format: {model_name: {column_name: description}}
        Returns number of successful updates.
        """
        try:
            if not file_path.exists():
                return 0
                
            with open(file_path, 'r') as f:
                data = self.yaml.load(f)
                
            if not data or 'models' not in data:
                return 0
                
            updated_count = 0
            
            for model in data['models']:
                m_name = model.get('name')
                if m_name in updates:
                    model_updates = updates[m_name]
                    
                    if 'columns' not in model:
                        model['columns'] = []
                        
                    # Create map of existing columns for faster lookup
                    existing_cols = {c.get('name'): c for c in model['columns']}
                    
                    for col_name, desc in model_updates.items():
                        if not desc:
                            continue
                            
                        if col_name in existing_cols:
                            if '\n' in desc:
                                existing_cols[col_name]['description'] = PreservedScalarString(desc)
                            else:
                                existing_cols[col_name]['description'] = desc
                        else:
                            new_col = {'name': col_name}
                            if '\n' in desc:
                                new_col['description'] = PreservedScalarString(desc)
                            else:
                                new_col['description'] = desc
                            model['columns'].append(new_col)
                            
                        updated_count += 1
            
            if updated_count > 0:
                with open(file_path, 'w') as f:
                    self.yaml.dump(data, f)
                    
            return updated_count
            
        except Exception as e:
            logger.error(f"Error in batch update for {file_path}: {e}")
            return 0
