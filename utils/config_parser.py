import yaml
import os

def load_config(config_paths):
    """Loads one or more YAML config files and merges them."""
    if isinstance(config_paths, str):
        config_paths = [config_paths]
    
    final_config = {}
    for path in config_paths:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        if '_base_' in config:
            base_path = os.path.join(os.path.dirname(path), config['_base_'])
            base_config = load_config(base_path)
            config = merge_dicts(base_config, config)
            config.pop('_base_')
            
        final_config = merge_dicts(final_config, config)
    
    return final_config

def merge_dicts(dict1, dict2):
    """Recursively merges dict2 into dict1."""
    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1 and isinstance(dict1[key], dict):
            merge_dicts(dict1[key], value)
        else:
            dict1[key] = value
    return dict1
