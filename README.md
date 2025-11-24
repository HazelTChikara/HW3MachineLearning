# Movie Recommender System - Task 2

This project implements and evaluates various recommender system algorithms on the MovieLens dataset.

## Author
HazelTChikara

## Date
November 23, 2025

## Description

This assignment implements three recommender system algorithms:
1. **Probabilistic Matrix Factorization (PMF)** - implemented using SVD
2. **User-based Collaborative Filtering**
3. **Item-based Collaborative Filtering**

The system evaluates these algorithms using MAE and RMSE metrics with 5-fold cross-validation.

## Dataset

Download `ratings_small.csv` from the MovieLens dataset:
- **Source**: [Kaggle - The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset)
- **File needed**: `ratings_small.csv`
- Place the file in the same directory as `recommender_system.py`

The dataset contains:
- User ratings for movies
- Format: `userId, movieId, rating, timestamp`
- Rating scale: 0.5 to 5.0

## Installation

### Step 1: Install Required Libraries

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-surprise scipy
```

### Step 2: Download the Dataset

1. Visit [Kaggle - The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset)
2. Download `ratings_small.csv`
3. Place it in the project directory

## Usage

Run the complete analysis:

```bash
python recommender_system.py
```

This will execute all tasks and generate the following output files:
- `task_2d_model_comparison.png` - Comparison of PMF, User-based CF, and Item-based CF
- `task_2e_similarity_metrics.png` - Impact of similarity metrics on CF algorithms
- `task_2f_neighbor_impact.png` - Impact of number of neighbors on CF performance
- `task_2g_best_k.png` - Identification of optimal k value

## Tasks Completed

### Task 2c: 5-Fold Cross-Validation
Computes average MAE and RMSE for:
- Probabilistic Matrix Factorization (PMF/SVD)
- User-based Collaborative Filtering (cosine similarity)
- Item-based Collaborative Filtering (cosine similarity)

### Task 2d: Model Comparison
Compares the three algorithms and identifies the best performing model based on:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Task 2e: Similarity Metrics Analysis
Examines the impact of three similarity metrics on CF algorithms:
- **Cosine similarity**
- **MSD (Mean Squared Difference)**
- **Pearson correlation**

Analyzes whether the impact is consistent between User-based and Item-based CF.

### Task 2f: Neighbor Count Analysis
Examines how the number of neighbors (k) impacts performance:
- Tests k values: 5, 10, 20, 30, 40, 50, 60, 70, 80
- Plots performance curves for both User-based and Item-based CF

### Task 2g: Optimal K Identification
Identifies the best k value for each CF method based on RMSE and determines:
- Whether the optimal k is the same for both methods
- The optimal neighborhood size for each algorithm

## Metrics

### MAE (Mean Absolute Error)
Measures the average magnitude of errors in predictions:
```
MAE = (1/n) × Σ|predicted - actual|
```

### RMSE (Root Mean Squared Error)
Measures the square root of average squared errors:
```
RMSE = sqrt((1/n) × Σ(predicted - actual)²)
```

## Dependencies

- **pandas**: Data manipulation and CSV reading
- **numpy**: Numerical computations
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **scikit-surprise**: Recommender system algorithms
- **scipy**: Scientific computing utilities

## Expected Output

The program will:
1. Load and display dataset statistics
2. Run 5-fold cross-validation for each algorithm
3. Generate comparison tables and visualizations
4. Save plots as PNG files
5. Print comprehensive analysis and conclusions

## Results Files

All generated plots are saved with high resolution (300 DPI) in the project directory:
- Model comparison charts
- Similarity metric impact analysis
- Neighbor count impact curves
- Best k value identification

## Notes

- The program uses the Surprise library's implementations of collaborative filtering
- PMF is implemented using SVD (Singular Value Decomposition)
- All results include standard deviations for statistical significance
- Cross-validation ensures robust performance estimates

## Troubleshooting

If you encounter issues:

1. **Import errors**: Ensure all dependencies are installed
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **File not found**: Make sure `ratings_small.csv` is in the same directory

3. **Memory issues**: The dataset is small enough for most systems, but if needed, reduce k_values in the code

## License

This project is for educational purposes as part of a Data Mining course assignment.
