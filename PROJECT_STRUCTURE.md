# Project Structure - Movie Recommender System

## 📁 Clean Project Organization

### Core Implementation Files
- **`recommender_system.py`** (23 KB) - Main implementation with all tasks (2c-2g)
- **`task_2c_solution.py`** (12 KB) - Standalone Task 2c with detailed comments
- **`recommender_system.ipynb`** (28 KB) - Interactive Jupyter notebook version
- **`generate_report.py`** (55 KB) - Script that generates the Word report

### Dataset
- **`ratings_small.csv`** (2.3 MB) - MovieLens dataset (100,004 ratings)
- **`archive/`** - Original dataset backup

### Generated Results
- **`2c.docx`** (940 KB) - **Final PDF-ready report with all tasks**
- **`task_2d_model_comparison.png`** (156 KB) - Algorithm comparison visualization
- **`task_2e_similarity_metrics.png`** (330 KB) - Similarity metrics analysis
- **`task_2f_neighbor_impact.png`** (192 KB) - Neighbor count impact visualization
- **`task_2g_best_k.png`** (429 KB) - Optimal K identification visualization

### Configuration & Setup
- **`requirements.txt`** (103 B) - Python dependencies
- **`setup.sh`** (1.2 KB) - Automated setup script
- **`activate_env.sh`** (964 B) - Quick environment activation
- **`venv_py311/`** - Python 3.11 virtual environment

### Documentation
- **`README.md`** (4.6 KB) - Complete project documentation
- **`IMPLEMENTATION_SUMMARY.md`** (7.6 KB) - Task breakdown with line numbers

## 🚀 Quick Start

1. **Activate environment:**
   ```bash
   source activate_env.sh
   ```

2. **Run full analysis:**
   ```bash
   python recommender_system.py
   ```

3. **View results:**
   - Open `2c.docx` for complete report
   - All visualizations generated as PNG files

## 📊 What Each Task Does

- **Task 2c:** Computes MAE & RMSE for PMF, User-CF, Item-CF with 5-fold CV
- **Task 2d:** Compares models and identifies PMF as best
- **Task 2e:** Analyzes Cosine, MSD, Pearson similarity impacts
- **Task 2f:** Tests k=[5,10,20,30,40,50,60,70,80] neighbor counts
- **Task 2g:** Identifies optimal K (User-CF: k=30, Item-CF: k=80)

## 📝 Key Results

| Algorithm | MAE | RMSE |
|-----------|-----|------|
| PMF (Best) | 0.6909 | 0.8964 |
| User-CF (k=30, MSD) | 0.7453 | 0.9697 |
| Item-CF (k=80, MSD) | 0.7211 | 0.9345 |

## 🔗 Repository

GitHub: https://github.com/HazelTChikara/titanic_assignment

---
*Last updated: November 23, 2025*
