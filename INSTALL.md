# Installation Guide for Text Humanizer

## Option 1: Install from Source

### Prerequisites
- Python 3.7 or newer
- pip package manager

### Steps

1. **Clone the repository** (if you haven't already)
   ```
   git clone https://github.com/yourusername/humanizer.git
   cd humanizer
   ```

2. **Install the package in development mode**
   ```
   pip install -e .
   ```
   
   This installs the package in "editable" mode, so changes to the source code will be reflected immediately.

3. **Download required language models**
   ```
   python -m spacy download en_core_web_sm
   ```

4. **Run the web application**
   ```
   streamlit run app.py
   ```

## Option 2: Install from Distribution Files

If you have a wheel file (.whl) or source distribution (.tar.gz):

