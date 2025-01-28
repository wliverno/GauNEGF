#!/bin/bash


# List of Python files to include
PYTHON_FILES=(
    "scf.py"
    "scfE.py"
    "density.py"
    "surfG1D.py"
    "surfGBethe.py"
    "surfGTester.py"
    "transport.py"
    "matTools.py"
    "fermiSearch.py"
)

# Create package directory if it doesn't exist
mkdir -p "gauNEGF"

# Create empty __init__.py if it doesn't exist
touch "gauNEGF/__init__.py"

# Create symlinks for each Python file
for file in "${PYTHON_FILES[@]}"; do
    if [ -f "$file" ]; then
        ln -sf "../$file" "gauNEGF/$file"
        echo "Created symlink for $file"
    else
        echo "Warning: $file not found"
    fi
done

# Install package in editable mode
pip install -e .

echo "Installation complete"
