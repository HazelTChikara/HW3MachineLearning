#!/bin/bash

# Activation script for the Python 3.11 virtual environment
# This ensures scikit-surprise runs correctly

echo "=========================================="
echo "Activating Python 3.11 Virtual Environment"
echo "=========================================="
echo ""

# Activate the virtual environment
source venv_py311/bin/activate

# Display Python version
echo "Python version:"
python --version
echo ""

# Display installed packages
echo "Key packages installed:"
pip list | grep -E "(scikit-surprise|pandas|numpy|matplotlib|seaborn|scipy)" || echo "Packages list"
echo ""

echo "=========================================="
echo "Environment Ready!"
echo "=========================================="
echo ""
echo "You can now:"
echo "1. Run the script: python recommender_system.py"
echo "2. Start Jupyter: jupyter notebook recommender_system.ipynb"
echo "3. Run Python interactively: python"
echo ""
echo "To deactivate later: deactivate"
echo ""
