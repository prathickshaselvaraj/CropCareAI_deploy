"""Utility to fix import paths"""
import sys
import os
from pathlib import Path

def setup_paths():
    """Setup Python paths for the project"""
    # Get project root (CropCareAI folder)
    project_root = Path(__file__).parent.parent.parent.parent
    src_path = project_root / "src"
    
    # Add to Python path if not already there
    paths_to_add = [str(project_root), str(src_path)]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
            print(f"✅ Added to path: {path}")
    
    return project_root, src_path

# Auto-setup when imported
project_root, src_path = setup_paths()