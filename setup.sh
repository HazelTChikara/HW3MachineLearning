#!/bin/bash

# Setup script for Movie Recommender System
# This script installs required dependencies

echo "=========================================="
echo "Movie Recommender System - Setup"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Install requirements
echo "Installing required Python packages..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Installation completed successfully!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Download 'ratings_small.csv' from:"
    echo "   https://www.kaggle.com/rounakbanik/the-movies-dataset"
    echo ""
    echo "2. Place the file in this directory"
    echo ""
    echo "3. Run the recommender system:"
    echo "   python3 recommender_system.py"
    echo ""
else
    echo ""
    echo "Installation failed. Please check the error messages above."
    exit 1
fi
