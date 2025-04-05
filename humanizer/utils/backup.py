"""
Utilities for backing up and restoring models and configuration.
"""
import os
import shutil
import zipfile
import tempfile
import json
import datetime
import glob
from pathlib import Path

def create_backup(source_dir, output_path=None, include_patterns=None):
    """
    Create a backup of the specified directory.
    
    Args:
        source_dir (str): Directory to back up
        output_path (str, optional): Custom output path for the backup file
        include_patterns (list, optional): File patterns to include (e.g., ['*.pkl', '*.json'])
        
    Returns:
        str: Path to the created backup file
    """
    source_path = Path(source_dir)
    
    # Default to backing up all model files
    if include_patterns is None:
        include_patterns = ['*.pkl']
    
    # Create a timestamp for the backup filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set default output path if none provided
    if output_path is None:
        output_path = source_path.parent / f"humanizer_backup_{timestamp}.zip"
    
    # Create temporary directory for organizing files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create metadata file
        metadata = {
            "created_at": timestamp,
            "source_directory": str(source_path),
            "file_count": 0,
            "version": "0.1.0"
        }
        
        # Copy all matching files to the temp directory
        file_count = 0
        for pattern in include_patterns:
            for file_path in source_path.glob(pattern):
                if file_path.is_file():
                    # Preserve directory structure relative to source
                    rel_path = file_path.relative_to(source_path)
                    dest_path = Path(temp_dir) / rel_path
                    
                    # Create parent directories if they don't exist
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy the file
                    shutil.copy2(file_path, dest_path)
                    file_count += 1
        
        # Update metadata with file count
        metadata["file_count"] = file_count
        
        # Write metadata file
        with open(os.path.join(temp_dir, "backup_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Create the ZIP file
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all files from temp directory
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Add file to zip with path relative to temp_dir
                    arc_name = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arc_name)
    
    print(f"Backup created: {output_path} with {file_count} files")
    return output_path

def restore_backup(backup_file, target_dir=None, overwrite=False):
    """
    Restore from a backup file.
    
    Args:
        backup_file (str): Path to the backup file
        target_dir (str, optional): Directory to restore to (default is original location)
        overwrite (bool): Whether to overwrite existing files
        
    Returns:
        bool: True if successful, False otherwise
    """
    backup_path = Path(backup_file)
    
    if not backup_path.exists():
        print(f"Error: Backup file {backup_file} not found")
        return False
    
    try:
        with zipfile.ZipFile(backup_path, 'r') as zipf:
            # Extract metadata first
            try:
                with zipf.open('backup_metadata.json') as f:
                    metadata = json.load(f)
            except (KeyError, json.JSONDecodeError):
                metadata = {"source_directory": None, "file_count": "unknown"}
            
            # Determine target directory
            if target_dir is None:
                if metadata["source_directory"]:
                    target_dir = metadata["source_directory"]
                else:
                    # Default to models directory if source not in metadata
                    target_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
            
            target_path = Path(target_dir)
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Extract all files
            for file_info in zipf.infolist():
                if file_info.filename == "backup_metadata.json":
                    continue  # Skip metadata file
                    
                target_file = target_path / file_info.filename
                
                # Check if file exists and handle according to overwrite setting
                if target_file.exists() and not overwrite:
                    print(f"Skipping {target_file} (already exists)")
                    continue
                
                # Create directories if needed
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Extract the file
                with zipf.open(file_info) as source, open(target_file, 'wb') as target:
                    shutil.copyfileobj(source, target)
            
            print(f"Backup restored to {target_path}")
            return True
    
    except Exception as e:
        print(f"Error restoring backup: {e}")
        return False
