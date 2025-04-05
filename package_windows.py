"""
Script to package the Text Humanizer application for Windows using PyInstaller.
"""
import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

def check_pyinstaller():
    """Check if PyInstaller is installed and install if necessary."""
    try:
        import PyInstaller
        print("PyInstaller is already installed.")
        return True
    except ImportError:
        print("PyInstaller not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
            print("PyInstaller installed successfully.")
            return True
        except Exception as e:
            print(f"Error installing PyInstaller: {e}")
            return False

def clean_build_dirs():
    """Clean up build directories."""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
                print(f"Removed {dir_name}/ directory")
            except Exception as e:
                print(f"Error cleaning {dir_name}: {e}")

def download_spacy_model():
    """Download the required spaCy model."""
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("spaCy model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading spaCy model: {e}")
        print("You might need to download it manually with: python -m spacy download en_core_web_sm")

def create_single_file_exe():
    """Create a single-file executable."""
    print("\nCreating single-file executable...")
    
    # Build command arguments list properly
    cmd = [
        "pyinstaller",
        "--name=text-humanizer",
        "--onefile",
        "--windowed",
        "--add-data=config;config",
        "--hidden-import=nltk",
        "--hidden-import=sklearn",
        "--hidden-import=language_tool_python",
        # Add package metadata to fix import errors
        "--copy-metadata=streamlit",
        "--copy-metadata=sklearn",
        "--copy-metadata=nltk",
        "--copy-metadata=spacy",
        # Collect all modules for streamlit
        "--collect-all=streamlit",
        "app.py"
    ]
    
    # Add icon only if it exists
    if os.path.exists("icon.ico"):
        cmd.insert(4, "--icon=icon.ico")
    
    try:
        subprocess.check_call(cmd)
        print("\n✅ Single-file executable created successfully in dist/text-humanizer.exe")
        return True
    except Exception as e:
        print(f"Error creating single-file executable: {e}")
        return False

def create_folder_distribution():
    """Create a folder distribution (more reliable but larger)."""
    print("\nCreating folder distribution...")
    
    # Build command arguments list properly
    cmd = [
        "pyinstaller",
        "--name=text-humanizer",
        "--add-data=config;config",
        "--hidden-import=nltk",
        "--hidden-import=sklearn",
        "--hidden-import=language_tool_python",
        # Add package metadata to fix import errors
        "--copy-metadata=streamlit",
        "--copy-metadata=sklearn",
        "--copy-metadata=nltk", 
        "--copy-metadata=spacy",
        # Collect all modules for streamlit
        "--collect-all=streamlit",
        "app.py"
    ]
    
    # Add icon only if it exists
    if os.path.exists("icon.ico"):
        cmd.insert(2, "--icon=icon.ico")
    
    try:
        subprocess.check_call(cmd)
        print("\n✅ Folder distribution created successfully in dist/text-humanizer/")
        return True
    except Exception as e:
        print(f"Error creating folder distribution: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Package Text Humanizer for Windows")
    parser.add_argument("--clean", action="store_true", help="Clean build directories before packaging")
    parser.add_argument("--type", choices=["onefile", "folder", "both"], default="both",
                        help="Type of distribution to create")
    
    args = parser.parse_args()
    
    print("===== Text Humanizer Windows Packaging Tool =====\n")
    
    # Install PyInstaller if not already installed
    if not check_pyinstaller():
        return
    
    # Clean up if requested
    if args.clean:
        clean_build_dirs()
    
    # Download spaCy model if needed
    download_spacy_model()
    
    # Create executable(s)
    if args.type in ["onefile", "both"]:
        create_single_file_exe()
    
    if args.type in ["folder", "both"]:
        create_folder_distribution()
    
    print("\n===== Packaging completed! =====")
    print("You can find the packaged application in the dist/ directory.")
    print("Note: The first run might take longer as it unpacks resources.")

if __name__ == "__main__":
    main()
