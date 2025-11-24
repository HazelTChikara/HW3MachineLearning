"""
Task 2c Solution: Compute Average MAE and RMSE with 5-Fold Cross-Validation
Author: HazelTChikara
Date: November 23, 2025

This script computes the average MAE and RMSE for three recommender algorithms:
1. Probabilistic Matrix Factorization (PMF)
2. User-based Collaborative Filtering
3. Item-based Collaborative Filtering

All evaluations use 5-fold cross-validation as required by the assignment.
"""

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import cross_validate


def compute_average_mae_rmse_with_5fold_cv(data):
    """
    TASK 2C: Compute average MAE and RMSE for PMF, User-based CF, and Item-based CF
    using 5-fold cross-validation.
    
    This function implements the core requirement for Task 2c:
    - Evaluates three different recommender system algorithms
    - Uses 5-fold cross-validation for robust performance estimation
    - Computes Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
    - Returns mean and standard deviation for each metric
    
    Parameters:
    -----------
    data : surprise.Dataset
        The movie ratings dataset loaded into Surprise format
        Expected columns: userId, movieId, rating
        
    Returns:
    --------
    dict : Dictionary containing results for each algorithm
        Structure: {
            'Algorithm_Name': {
                'MAE': float,        # Mean Absolute Error (average across 5 folds)
                'RMSE': float,       # Root Mean Squared Error (average across 5 folds)
                'MAE_std': float,    # Standard deviation of MAE
                'RMSE_std': float    # Standard deviation of RMSE
            }
        }
    
    Algorithm Details:
    ------------------
    1. PMF (Probabilistic Matrix Factorization):
       - Implemented using SVD (Singular Value Decomposition)
       - Matrix factorization approach
       - Learns latent factors for users and items
       
    2. User-based Collaborative Filtering:
       - Finds similar users based on rating patterns
       - Predicts ratings based on similar users' preferences
       - Uses cosine similarity by default
       
    3. Item-based Collaborative Filtering:
       - Finds similar items based on user ratings
       - Predicts ratings based on similar items
       - Uses cosine similarity by default
    
    Evaluation Metrics:
    -------------------
    MAE (Mean Absolute Error):
        - Average absolute difference between predicted and actual ratings
        - Formula: MAE = (1/n) × Σ|predicted - actual|
        - Lower is better
        
    RMSE (Root Mean Squared Error):
        - Square root of average squared differences
        - Formula: RMSE = sqrt((1/n) × Σ(predicted - actual)²)
        - Penalizes larger errors more than MAE
        - Lower is better
        
    Cross-Validation:
    -----------------
    5-fold cross-validation divides the data into 5 subsets:
    - Each fold uses 80% data for training, 20% for testing
    - Results are averaged across all 5 folds
    - Provides robust performance estimate
    """
    
    # Initialize results dictionary to store all metrics
    results = {}
    
    # ============================================================================
    # ALGORITHM 1: Probabilistic Matrix Factorization (PMF)
    # ============================================================================
    print("\n" + "="*80)
    print("ALGORITHM 1: Probabilistic Matrix Factorization (PMF using SVD)")
    print("="*80)
    
    # Create PMF model using SVD algorithm
    # SVD learns latent factors by decomposing the user-item rating matrix
    pmf_model = SVD()
    
    # Perform 5-fold cross-validation
    # cv=5 means split data into 5 folds
    # measures=['MAE', 'RMSE'] specifies which metrics to compute
    # verbose=True prints detailed results for each fold
    print("Running 5-fold cross-validation for PMF...")
    pmf_cv_results = cross_validate(
        pmf_model, 
        data, 
        measures=['MAE', 'RMSE'],  # Compute both MAE and RMSE
        cv=5,                        # Use 5 folds
        verbose=True                 # Show results for each fold
    )
    
    # Calculate average MAE and RMSE across all 5 folds
    # test_mae is a numpy array with MAE for each of the 5 folds
    pmf_avg_mae = pmf_cv_results['test_mae'].mean()
    pmf_avg_rmse = pmf_cv_results['test_rmse'].mean()
    
    # Calculate standard deviation to measure consistency
    pmf_std_mae = pmf_cv_results['test_mae'].std()
    pmf_std_rmse = pmf_cv_results['test_rmse'].std()
    
    # Store results in dictionary
    results['PMF'] = {
        'MAE': pmf_avg_mae,
        'RMSE': pmf_avg_rmse,
        'MAE_std': pmf_std_mae,
        'RMSE_std': pmf_std_rmse
    }
    
    print(f"\nPMF Results:")
    print(f"  Average MAE:  {pmf_avg_mae:.4f} ± {pmf_std_mae:.4f}")
    print(f"  Average RMSE: {pmf_avg_rmse:.4f} ± {pmf_std_rmse:.4f}")
    
    # ============================================================================
    # ALGORITHM 2: User-based Collaborative Filtering
    # ============================================================================
    print("\n" + "="*80)
    print("ALGORITHM 2: User-based Collaborative Filtering")
    print("="*80)
    
    # Create User-based CF model
    # sim_options configures the similarity metric
    # 'name': 'cosine' uses cosine similarity to find similar users
    # 'user_based': True means we're finding similar users (not items)
    user_cf_model = KNNBasic(
        sim_options={
            'name': 'cosine',      # Use cosine similarity
            'user_based': True     # User-based (not item-based)
        }
    )
    
    # Perform 5-fold cross-validation for User-based CF
    print("Running 5-fold cross-validation for User-based CF...")
    user_cv_results = cross_validate(
        user_cf_model,
        data,
        measures=['MAE', 'RMSE'],
        cv=5,
        verbose=True
    )
    
    # Calculate average MAE and RMSE across all 5 folds
    user_avg_mae = user_cv_results['test_mae'].mean()
    user_avg_rmse = user_cv_results['test_rmse'].mean()
    
    # Calculate standard deviation
    user_std_mae = user_cv_results['test_mae'].std()
    user_std_rmse = user_cv_results['test_rmse'].std()
    
    # Store results
    results['User-based CF'] = {
        'MAE': user_avg_mae,
        'RMSE': user_avg_rmse,
        'MAE_std': user_std_mae,
        'RMSE_std': user_std_rmse
    }
    
    print(f"\nUser-based CF Results:")
    print(f"  Average MAE:  {user_avg_mae:.4f} ± {user_std_mae:.4f}")
    print(f"  Average RMSE: {user_avg_rmse:.4f} ± {user_std_rmse:.4f}")
    
    # ============================================================================
    # ALGORITHM 3: Item-based Collaborative Filtering
    # ============================================================================
    print("\n" + "="*80)
    print("ALGORITHM 3: Item-based Collaborative Filtering")
    print("="*80)
    
    # Create Item-based CF model
    # 'user_based': False means we're finding similar items (not users)
    item_cf_model = KNNBasic(
        sim_options={
            'name': 'cosine',      # Use cosine similarity
            'user_based': False    # Item-based (not user-based)
        }
    )
    
    # Perform 5-fold cross-validation for Item-based CF
    print("Running 5-fold cross-validation for Item-based CF...")
    item_cv_results = cross_validate(
        item_cf_model,
        data,
        measures=['MAE', 'RMSE'],
        cv=5,
        verbose=True
    )
    
    # Calculate average MAE and RMSE across all 5 folds
    item_avg_mae = item_cv_results['test_mae'].mean()
    item_avg_rmse = item_cv_results['test_rmse'].mean()
    
    # Calculate standard deviation
    item_std_mae = item_cv_results['test_mae'].std()
    item_std_rmse = item_cv_results['test_rmse'].std()
    
    # Store results
    results['Item-based CF'] = {
        'MAE': item_avg_mae,
        'RMSE': item_avg_rmse,
        'MAE_std': item_std_mae,
        'RMSE_std': item_std_rmse
    }
    
    print(f"\nItem-based CF Results:")
    print(f"  Average MAE:  {item_avg_mae:.4f} ± {item_std_mae:.4f}")
    print(f"  Average RMSE: {item_avg_rmse:.4f} ± {item_std_rmse:.4f}")
    
    # ============================================================================
    # SUMMARY: Display all results in a comparison table
    # ============================================================================
    print("\n" + "="*80)
    print("TASK 2C SUMMARY: Average MAE and RMSE (5-Fold Cross-Validation)")
    print("="*80)
    print(f"{'Algorithm':<30} {'MAE':<20} {'RMSE':<20}")
    print("-"*80)
    
    for algo_name, metrics in results.items():
        mae_str = f"{metrics['MAE']:.4f} ± {metrics['MAE_std']:.4f}"
        rmse_str = f"{metrics['RMSE']:.4f} ± {metrics['RMSE_std']:.4f}"
        print(f"{algo_name:<30} {mae_str:<20} {rmse_str:<20}")
    
    print("="*80)
    print("\nInterpretation:")
    print("- Lower MAE/RMSE indicates better prediction accuracy")
    print("- Standard deviation shows consistency across folds")
    print("- All values represent averages across 5 independent test sets")
    print("="*80)
    
    return results


def load_ratings_data(filepath='ratings_small.csv'):
    """
    Load the movie ratings dataset and prepare it for Surprise library.
    
    Parameters:
    -----------
    filepath : str
        Path to the ratings_small.csv file
        Expected format: userId, movieId, rating, timestamp
        
    Returns:
    --------
    surprise.Dataset
        Dataset object ready for use with Surprise algorithms
    """
    print("="*80)
    print("LOADING DATASET")
    print("="*80)
    
    # Read CSV file into pandas DataFrame
    df = pd.read_csv(filepath)
    
    # Display basic information
    print(f"Dataset loaded: {filepath}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Display dataset statistics
    print(f"\nDataset Statistics:")
    print(f"  Number of users:   {df['userId'].nunique()}")
    print(f"  Number of movies:  {df['movieId'].nunique()}")
    print(f"  Number of ratings: {len(df)}")
    print(f"  Rating range:      [{df['rating'].min()}, {df['rating'].max()}]")
    print(f"  Average rating:    {df['rating'].mean():.2f}")
    print(f"  Sparsity:          {(1 - len(df) / (df['userId'].nunique() * df['movieId'].nunique())) * 100:.2f}%")
    
    # Create Surprise Reader object
    # rating_scale specifies the range of possible ratings
    reader = Reader(rating_scale=(0.5, 5.0))
    
    # Load data into Surprise Dataset format
    # Only need userId, movieId, and rating columns
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    
    print("\n✓ Dataset successfully loaded and prepared for Surprise library")
    print("="*80)
    
    return data


def main():
    """
    Main function to execute Task 2c.
    
    This function:
    1. Loads the movie ratings dataset
    2. Computes average MAE and RMSE for three algorithms
    3. Uses 5-fold cross-validation for robust evaluation
    4. Displays results in a clear comparison table
    """
    print("\n" + "="*80)
    print("TASK 2C: RECOMMENDER SYSTEM EVALUATION")
    print("Computing Average MAE and RMSE with 5-Fold Cross-Validation")
    print("="*80 + "\n")
    
    # Step 1: Load the dataset
    data = load_ratings_data('ratings_small.csv')
    
    # Step 2: Compute MAE and RMSE for all three algorithms
    results = compute_average_mae_rmse_with_5fold_cv(data)
    
    # Step 3: Create results DataFrame for easy viewing
    print("\n" + "="*80)
    print("FINAL RESULTS (Task 2c)")
    print("="*80)
    
    results_df = pd.DataFrame(results).T
    print("\nResults DataFrame:")
    print(results_df[['MAE', 'RMSE']])
    
    print("\n" + "="*80)
    print("✓ TASK 2C COMPLETED SUCCESSFULLY")
    print("="*80)
    
    return results


# ============================================================================
# EXECUTE THE SCRIPT
# ============================================================================
if __name__ == "__main__":
    # Run the complete Task 2c analysis
    results = main()
    
    # Results are stored in the 'results' dictionary
    # Can be accessed as: results['PMF']['MAE'], results['User-based CF']['RMSE'], etc.
