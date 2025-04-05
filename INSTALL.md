# Installing Text Humanizer

## Option 1: Install as a Python Package

### Prerequisites
- Python 3.7 or newer
- pip package manager

### Installation Steps

1. Install the package directly from the source directory:
   ```
   pip install -e .
   ```

2. Download required language models:
   ```
   python -m spacy download en_core_web_sm
   ```

3. Run the web application:
   ```
   humanize-text --web
   ```
   
   Or use the command directly:
   ```
   streamlit run app.py
   ```

## Option 2: Run the Windows Executable

If you have the pre-built executable:

1. Simply double-click `text-humanizer.exe` to run the application
2. The web interface will open in your default browser

## Option 3: Build Your Own Windows Executable

1. Make sure PyInstaller is installed:
   ```
   pip install pyinstaller
   ```

2. Run the packaging script:
   ```
   python package_windows.py --clean
   ```
   
   Optional arguments:
   - `--type onefile`: Create only a single executable file
   - `--type folder`: Create only a folder distribution (more reliable)
   - `--clean`: Clean build directories before packaging

3. Find the executable in the `dist` folder:
   - Single file: `dist/text-humanizer.exe`
   - Folder distribution: `dist/text-humanizer/text-humanizer.exe`

4. Custom icon (optional):
   - Place an `icon.ico` file in the project root directory
   - Run the packaging script again

## Command-line Usage

After installing as a Python package, you can use Text Humanizer from the command line:

