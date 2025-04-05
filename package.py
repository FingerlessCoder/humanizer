"""
Script to package the Humanizer application.
"""
import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

def clean_build_dirs():
    """Remove build directories."""
    dirs_to_clean = ['build', 'dist', 'humanizer.egg-info']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Removed {dir_name}/ directory")

def build_package(dist_type="all"):
    """Build the Python package."""
    print("Building the humanizer package...")
    
    if dist_type in ["all", "source"]:
        # Build source distribution
        subprocess.check_call([sys.executable, "setup.py", "sdist"])
        print("Source distribution created.")
    
    if dist_type in ["all", "wheel"]:
        # Build wheel distribution
        try:
            subprocess.check_call([sys.executable, "setup.py", "bdist_wheel"])
            print("Wheel distribution created.")
        except subprocess.CalledProcessError:
            print("Wheel build failed. Make sure wheel package is installed: pip install wheel")
            if dist_type == "wheel":
                return False
    
    return True

def create_executable():
    """Create standalone executable using PyInstaller."""
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    print("Creating standalone executable...")
    subprocess.check_call([
        "pyinstaller",
        "--name=text-humanizer",
        "--onefile",
        "--windowed",  # Remove this if you want a console window
        "--add-data=config;config",
        "app.py"
    ])
    print("Executable created in dist/ directory")
    return True

def main():
    parser = argparse.ArgumentParser(description="Package the Humanizer application")
    parser.add_argument("--clean", action="store_true", help="Clean build directories before packaging")
    parser.add_argument("--type", choices=["all", "source", "wheel", "exe"], default="all",
                        help="Type of package to build")
    
    args = parser.parse_args()
    
    if args.clean:
        clean_build_dirs()
    
    if args.type == "exe":
        create_executable()
    else:
        build_package(args.type)
    
    print("Packaging completed successfully!")

if __name__ == "__main__":
    main()
