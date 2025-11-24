"""
Recommender System Implementation for Movie Ratings
Author: HazelTChikara
Date: November 23, 2025

This script implements and evaluates various recommender system algorithms:
- Probabilistic Matrix Factorization (PMF)
- User-based Collaborative Filtering
- Item-based Collaborative Filtering

Dataset: ratings_small.csv from MovieLens
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import cross_validate
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_data(filepath='ratings_small.csv'):
    """
    Load the ratings data from CSV file
    
    Args:
        filepath: Path to the ratings_small.csv file
        
    Returns:
        Surprise Dataset object
    """
    print("Loading data from:", filepath)
    
    # Read the CSV file
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Basic statistics
    print(f"\nDataset Statistics:")
    print(f"Number of users: {df['userId'].nunique()}")
    print(f"Number of movies: {df['movieId'].nunique()}")
    print(f"Number of ratings: {len(df)}")
    print(f"Rating range: [{df['rating'].min()}, {df['rating'].max()}]")
    print(f"Average rating: {df['rating'].mean():.2f}")
    
    # Create Surprise dataset
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    
    return data


def task_2c_basic_comparison(data):
    """
    Task 2c: Compute average MAE and RMSE for PMF, User-based CF, and Item-based CF
    using 5-fold cross-validation
    
    Args:
        data: Surprise Dataset object
        
    Returns:
        Dictionary with results for each algorithm
    """
    print("\n" + "="*80)
    print("TASK 2C: 5-Fold Cross-Validation Comparison")
    print("="*80)
    
    results = {}
    
    # 1. Probabilistic Matrix Factorization (PMF) - using SVD as implementation
    print("\n1. Probabilistic Matrix Factorization (SVD)...")
    pmf = SVD()
    pmf_results = cross_validate(pmf, data, measures=['MAE', 'RMSE'], cv=5, verbose=True)
    results['PMF'] = {
        'MAE': pmf_results['test_mae'].mean(),
        'RMSE': pmf_results['test_rmse'].mean(),
        'MAE_std': pmf_results['test_mae'].std(),
        'RMSE_std': pmf_results['test_rmse'].std()
    }
    
    # 2. User-based Collaborative Filtering
    print("\n2. User-based Collaborative Filtering...")
    user_cf = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
    user_results = cross_validate(user_cf, data, measures=['MAE', 'RMSE'], cv=5, verbose=True)
    results['User-based CF'] = {
        'MAE': user_results['test_mae'].mean(),
        'RMSE': user_results['test_rmse'].mean(),
        'MAE_std': user_results['test_mae'].std(),
        'RMSE_std': user_results['test_rmse'].std()
    }
    
    # 3. Item-based Collaborative Filtering
    print("\n3. Item-based Collaborative Filtering...")
    item_cf = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
    item_results = cross_validate(item_cf, data, measures=['MAE', 'RMSE'], cv=5, verbose=True)
    results['Item-based CF'] = {
        'MAE': item_results['test_mae'].mean(),
        'RMSE': item_results['test_rmse'].mean(),
        'MAE_std': item_results['test_mae'].std(),
        'RMSE_std': item_results['test_rmse'].std()
    }
    
    # Print summary
    print("\n" + "-"*80)
    print("SUMMARY OF RESULTS:")
    print("-"*80)
    print(f"{'Algorithm':<25} {'MAE':<15} {'RMSE':<15}")
    print("-"*80)
    for algo_name, metrics in results.items():
        print(f"{algo_name:<25} {metrics['MAE']:.4f} ± {metrics['MAE_std']:.4f}   {metrics['RMSE']:.4f} ± {metrics['RMSE_std']:.4f}")
    print("-"*80)
    
    return results


def task_2d_compare_models(results):
    """
    Task 2d: Compare model performances and identify the best model
    
    Args:
        results: Dictionary with results from task_2c
    """
    print("\n" + "="*80)
    print("TASK 2D: Model Comparison and Best Model Selection")
    print("="*80)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results).T
    print("\nComparison Table:")
    print(comparison_df[['MAE', 'RMSE']])
    
    # Find best model
    best_mae_model = comparison_df['MAE'].idxmin()
    best_rmse_model = comparison_df['RMSE'].idxmin()
    
    print(f"\nBest Model by MAE: {best_mae_model} (MAE = {comparison_df.loc[best_mae_model, 'MAE']:.4f})")
    print(f"Best Model by RMSE: {best_rmse_model} (RMSE = {comparison_df.loc[best_rmse_model, 'RMSE']:.4f})")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # MAE comparison
    algorithms = list(results.keys())
    mae_values = [results[algo]['MAE'] for algo in algorithms]
    mae_std = [results[algo]['MAE_std'] for algo in algorithms]
    
    axes[0].bar(algorithms, mae_values, yerr=mae_std, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_ylabel('MAE', fontsize=12)
    axes[0].set_title('Mean Absolute Error (MAE) Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim(bottom=0)
    axes[0].tick_params(axis='x', rotation=15)
    
    # Add value labels on bars
    for i, (v, s) in enumerate(zip(mae_values, mae_std)):
        axes[0].text(i, v + s + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    
    # RMSE comparison
    rmse_values = [results[algo]['RMSE'] for algo in algorithms]
    rmse_std = [results[algo]['RMSE_std'] for algo in algorithms]
    
    axes[1].bar(algorithms, rmse_values, yerr=rmse_std, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].set_title('Root Mean Squared Error (RMSE) Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylim(bottom=0)
    axes[1].tick_params(axis='x', rotation=15)
    
    # Add value labels on bars
    for i, (v, s) in enumerate(zip(rmse_values, rmse_std)):
        axes[1].text(i, v + s + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('task_2d_model_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'task_2d_model_comparison.png'")
    plt.show()
    
    # Conclusion
    print("\n" + "-"*80)
    print("CONCLUSION:")
    print("-"*80)
    if best_mae_model == best_rmse_model:
        print(f"The best model is {best_mae_model}, which achieves the lowest")
        print(f"error in both MAE ({comparison_df.loc[best_mae_model, 'MAE']:.4f}) and RMSE ({comparison_df.loc[best_rmse_model, 'RMSE']:.4f}).")
    else:
        print(f"The results vary by metric:")
        print(f"- Best by MAE: {best_mae_model} (MAE = {comparison_df.loc[best_mae_model, 'MAE']:.4f})")
        print(f"- Best by RMSE: {best_rmse_model} (RMSE = {comparison_df.loc[best_rmse_model, 'RMSE']:.4f})")
    print("-"*80)


def task_2e_similarity_metrics(data):
    """
    Task 2e: Examine impact of cosine, MSD, and Pearson similarities on CF algorithms
    
    Args:
        data: Surprise Dataset object
    """
    print("\n" + "="*80)
    print("TASK 2E: Impact of Similarity Metrics on Collaborative Filtering")
    print("="*80)
    
    similarities = ['cosine', 'msd', 'pearson']
    user_results = defaultdict(dict)
    item_results = defaultdict(dict)
    
    for sim in similarities:
        print(f"\n--- Testing similarity: {sim.upper()} ---")
        
        # User-based CF
        print(f"User-based CF with {sim}...")
        user_cf = KNNBasic(sim_options={'name': sim, 'user_based': True})
        user_cv = cross_validate(user_cf, data, measures=['MAE', 'RMSE'], cv=5, verbose=False)
        user_results[sim]['MAE'] = user_cv['test_mae'].mean()
        user_results[sim]['RMSE'] = user_cv['test_rmse'].mean()
        user_results[sim]['MAE_std'] = user_cv['test_mae'].std()
        user_results[sim]['RMSE_std'] = user_cv['test_rmse'].std()
        print(f"  MAE: {user_results[sim]['MAE']:.4f}, RMSE: {user_results[sim]['RMSE']:.4f}")
        
        # Item-based CF
        print(f"Item-based CF with {sim}...")
        item_cf = KNNBasic(sim_options={'name': sim, 'user_based': False})
        item_cv = cross_validate(item_cf, data, measures=['MAE', 'RMSE'], cv=5, verbose=False)
        item_results[sim]['MAE'] = item_cv['test_mae'].mean()
        item_results[sim]['RMSE'] = item_cv['test_rmse'].mean()
        item_results[sim]['MAE_std'] = item_cv['test_mae'].std()
        item_results[sim]['RMSE_std'] = item_cv['test_rmse'].std()
        print(f"  MAE: {item_results[sim]['MAE']:.4f}, RMSE: {item_results[sim]['RMSE']:.4f}")
    
    # Print summary table
    print("\n" + "-"*80)
    print("SUMMARY TABLE:")
    print("-"*80)
    print("\nUser-based CF:")
    print(f"{'Similarity':<15} {'MAE':<20} {'RMSE':<20}")
    print("-"*80)
    for sim in similarities:
        print(f"{sim.upper():<15} {user_results[sim]['MAE']:.4f} ± {user_results[sim]['MAE_std']:.4f}    {user_results[sim]['RMSE']:.4f} ± {user_results[sim]['RMSE_std']:.4f}")
    
    print("\nItem-based CF:")
    print(f"{'Similarity':<15} {'MAE':<20} {'RMSE':<20}")
    print("-"*80)
    for sim in similarities:
        print(f"{sim.upper():<15} {item_results[sim]['MAE']:.4f} ± {item_results[sim]['MAE_std']:.4f}    {item_results[sim]['RMSE']:.4f} ± {item_results[sim]['RMSE_std']:.4f}")
    print("-"*80)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    x_pos = np.arange(len(similarities))
    width = 0.35
    
    # MAE comparison
    user_mae = [user_results[sim]['MAE'] for sim in similarities]
    item_mae = [item_results[sim]['MAE'] for sim in similarities]
    user_mae_std = [user_results[sim]['MAE_std'] for sim in similarities]
    item_mae_std = [item_results[sim]['MAE_std'] for sim in similarities]
    
    axes[0, 0].bar(x_pos - width/2, user_mae, width, yerr=user_mae_std, label='User-based CF', 
                   capsize=5, alpha=0.8, color='#1f77b4')
    axes[0, 0].bar(x_pos + width/2, item_mae, width, yerr=item_mae_std, label='Item-based CF', 
                   capsize=5, alpha=0.8, color='#ff7f0e')
    axes[0, 0].set_ylabel('MAE', fontsize=12)
    axes[0, 0].set_title('MAE: User-based vs Item-based CF', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels([s.upper() for s in similarities])
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # RMSE comparison
    user_rmse = [user_results[sim]['RMSE'] for sim in similarities]
    item_rmse = [item_results[sim]['RMSE'] for sim in similarities]
    user_rmse_std = [user_results[sim]['RMSE_std'] for sim in similarities]
    item_rmse_std = [item_results[sim]['RMSE_std'] for sim in similarities]
    
    axes[0, 1].bar(x_pos - width/2, user_rmse, width, yerr=user_rmse_std, label='User-based CF', 
                   capsize=5, alpha=0.8, color='#1f77b4')
    axes[0, 1].bar(x_pos + width/2, item_rmse, width, yerr=item_rmse_std, label='Item-based CF', 
                   capsize=5, alpha=0.8, color='#ff7f0e')
    axes[0, 1].set_ylabel('RMSE', fontsize=12)
    axes[0, 1].set_title('RMSE: User-based vs Item-based CF', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels([s.upper() for s in similarities])
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # User-based CF only
    axes[1, 0].plot(similarities, user_mae, 'o-', label='MAE', linewidth=2, markersize=8, color='#2ca02c')
    axes[1, 0].plot(similarities, user_rmse, 's-', label='RMSE', linewidth=2, markersize=8, color='#d62728')
    axes[1, 0].set_xlabel('Similarity Metric', fontsize=12)
    axes[1, 0].set_ylabel('Error', fontsize=12)
    axes[1, 0].set_title('User-based CF: Similarity Impact', fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Item-based CF only
    axes[1, 1].plot(similarities, item_mae, 'o-', label='MAE', linewidth=2, markersize=8, color='#2ca02c')
    axes[1, 1].plot(similarities, item_rmse, 's-', label='RMSE', linewidth=2, markersize=8, color='#d62728')
    axes[1, 1].set_xlabel('Similarity Metric', fontsize=12)
    axes[1, 1].set_ylabel('Error', fontsize=12)
    axes[1, 1].set_title('Item-based CF: Similarity Impact', fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task_2e_similarity_metrics.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'task_2e_similarity_metrics.png'")
    plt.show()
    
    # Analysis
    print("\n" + "-"*80)
    print("ANALYSIS:")
    print("-"*80)
    
    # Check consistency
    user_best_mae = min(similarities, key=lambda s: user_results[s]['MAE'])
    user_best_rmse = min(similarities, key=lambda s: user_results[s]['RMSE'])
    item_best_mae = min(similarities, key=lambda s: item_results[s]['MAE'])
    item_best_rmse = min(similarities, key=lambda s: item_results[s]['RMSE'])
    
    print(f"Best similarity for User-based CF: {user_best_mae.upper()} (MAE), {user_best_rmse.upper()} (RMSE)")
    print(f"Best similarity for Item-based CF: {item_best_mae.upper()} (MAE), {item_best_rmse.upper()} (RMSE)")
    
    if user_best_mae == item_best_mae and user_best_rmse == item_best_rmse:
        print(f"\nThe impact of similarity metrics is CONSISTENT between User-based and Item-based CF.")
        print(f"Both methods perform best with {user_best_mae.upper()} similarity.")
    else:
        print(f"\nThe impact of similarity metrics is NOT CONSISTENT between User-based and Item-based CF.")
        print("Different similarity metrics work better for each method.")
    print("-"*80)
    
    return user_results, item_results


def task_2f_neighbor_impact(data):
    """
    Task 2f: Examine how the number of neighbors impacts CF performance
    
    Args:
        data: Surprise Dataset object
    """
    print("\n" + "="*80)
    print("TASK 2F: Impact of Number of Neighbors on Collaborative Filtering")
    print("="*80)
    
    # Test different k values
    k_values = [5, 10, 20, 30, 40, 50, 60, 70, 80]
    user_results = defaultdict(dict)
    item_results = defaultdict(dict)
    
    for k in k_values:
        print(f"\n--- Testing k = {k} ---")
        
        # User-based CF
        print(f"User-based CF with k={k}...")
        user_cf = KNNBasic(k=k, sim_options={'name': 'cosine', 'user_based': True})
        user_cv = cross_validate(user_cf, data, measures=['MAE', 'RMSE'], cv=5, verbose=False)
        user_results[k]['MAE'] = user_cv['test_mae'].mean()
        user_results[k]['RMSE'] = user_cv['test_rmse'].mean()
        user_results[k]['MAE_std'] = user_cv['test_mae'].std()
        user_results[k]['RMSE_std'] = user_cv['test_rmse'].std()
        print(f"  MAE: {user_results[k]['MAE']:.4f}, RMSE: {user_results[k]['RMSE']:.4f}")
        
        # Item-based CF
        print(f"Item-based CF with k={k}...")
        item_cf = KNNBasic(k=k, sim_options={'name': 'cosine', 'user_based': False})
        item_cv = cross_validate(item_cf, data, measures=['MAE', 'RMSE'], cv=5, verbose=False)
        item_results[k]['MAE'] = item_cv['test_mae'].mean()
        item_results[k]['RMSE'] = item_cv['test_rmse'].mean()
        item_results[k]['MAE_std'] = item_cv['test_mae'].std()
        item_results[k]['RMSE_std'] = item_cv['test_rmse'].std()
        print(f"  MAE: {item_results[k]['MAE']:.4f}, RMSE: {item_results[k]['RMSE']:.4f}")
    
    # Print summary table
    print("\n" + "-"*80)
    print("SUMMARY TABLE:")
    print("-"*80)
    print("\nUser-based CF:")
    print(f"{'k':<10} {'MAE':<20} {'RMSE':<20}")
    print("-"*80)
    for k in k_values:
        print(f"{k:<10} {user_results[k]['MAE']:.4f} ± {user_results[k]['MAE_std']:.4f}    {user_results[k]['RMSE']:.4f} ± {user_results[k]['RMSE_std']:.4f}")
    
    print("\nItem-based CF:")
    print(f"{'k':<10} {'MAE':<20} {'RMSE':<20}")
    print("-"*80)
    for k in k_values:
        print(f"{k:<10} {item_results[k]['MAE']:.4f} ± {item_results[k]['MAE_std']:.4f}    {item_results[k]['RMSE']:.4f} ± {item_results[k]['RMSE_std']:.4f}")
    print("-"*80)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data for plotting
    user_mae = [user_results[k]['MAE'] for k in k_values]
    user_rmse = [user_results[k]['RMSE'] for k in k_values]
    item_mae = [item_results[k]['MAE'] for k in k_values]
    item_rmse = [item_results[k]['RMSE'] for k in k_values]
    
    # User-based CF
    ax1 = axes[0]
    ax1.plot(k_values, user_mae, 'o-', label='MAE', linewidth=2, markersize=8, color='#2ca02c')
    ax1.plot(k_values, user_rmse, 's-', label='RMSE', linewidth=2, markersize=8, color='#d62728')
    ax1.set_xlabel('Number of Neighbors (k)', fontsize=12)
    ax1.set_ylabel('Error', fontsize=12)
    ax1.set_title('User-based CF: Impact of k', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(k_values)
    
    # Item-based CF
    ax2 = axes[1]
    ax2.plot(k_values, item_mae, 'o-', label='MAE', linewidth=2, markersize=8, color='#2ca02c')
    ax2.plot(k_values, item_rmse, 's-', label='RMSE', linewidth=2, markersize=8, color='#d62728')
    ax2.set_xlabel('Number of Neighbors (k)', fontsize=12)
    ax2.set_ylabel('Error', fontsize=12)
    ax2.set_title('Item-based CF: Impact of k', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(k_values)
    
    plt.tight_layout()
    plt.savefig('task_2f_neighbor_impact.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'task_2f_neighbor_impact.png'")
    plt.show()
    
    return user_results, item_results, k_values


def task_2g_best_k(user_results, item_results, k_values):
    """
    Task 2g: Identify the best k for User/Item based CF
    
    Args:
        user_results: Results from task_2f for user-based CF
        item_results: Results from task_2f for item-based CF
        k_values: List of k values tested
    """
    print("\n" + "="*80)
    print("TASK 2G: Identifying Best K Value")
    print("="*80)
    
    # Find best k for each method based on RMSE
    user_best_k = min(k_values, key=lambda k: user_results[k]['RMSE'])
    item_best_k = min(k_values, key=lambda k: item_results[k]['RMSE'])
    
    print("\nBest K based on RMSE:")
    print(f"User-based CF: k = {user_best_k} (RMSE = {user_results[user_best_k]['RMSE']:.4f})")
    print(f"Item-based CF: k = {item_best_k} (RMSE = {item_results[item_best_k]['RMSE']:.4f})")
    
    # Also show best k based on MAE
    user_best_k_mae = min(k_values, key=lambda k: user_results[k]['MAE'])
    item_best_k_mae = min(k_values, key=lambda k: item_results[k]['MAE'])
    
    print("\nBest K based on MAE:")
    print(f"User-based CF: k = {user_best_k_mae} (MAE = {user_results[user_best_k_mae]['MAE']:.4f})")
    print(f"Item-based CF: k = {item_best_k_mae} (MAE = {item_results[item_best_k_mae]['MAE']:.4f})")
    
    # Comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # RMSE comparison highlighting best k
    user_rmse = [user_results[k]['RMSE'] for k in k_values]
    item_rmse = [item_results[k]['RMSE'] for k in k_values]
    
    axes[0, 0].plot(k_values, user_rmse, 'o-', linewidth=2, markersize=8, color='#1f77b4', label='User-based CF')
    axes[0, 0].plot([user_best_k], [user_results[user_best_k]['RMSE']], 'r*', 
                     markersize=20, label=f'Best k={user_best_k}')
    axes[0, 0].set_xlabel('Number of Neighbors (k)', fontsize=12)
    axes[0, 0].set_ylabel('RMSE', fontsize=12)
    axes[0, 0].set_title('User-based CF: RMSE vs k', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_xticks(k_values)
    
    axes[0, 1].plot(k_values, item_rmse, 'o-', linewidth=2, markersize=8, color='#ff7f0e', label='Item-based CF')
    axes[0, 1].plot([item_best_k], [item_results[item_best_k]['RMSE']], 'r*', 
                     markersize=20, label=f'Best k={item_best_k}')
    axes[0, 1].set_xlabel('Number of Neighbors (k)', fontsize=12)
    axes[0, 1].set_ylabel('RMSE', fontsize=12)
    axes[0, 1].set_title('Item-based CF: RMSE vs k', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_xticks(k_values)
    
    # MAE comparison highlighting best k
    user_mae = [user_results[k]['MAE'] for k in k_values]
    item_mae = [item_results[k]['MAE'] for k in k_values]
    
    axes[1, 0].plot(k_values, user_mae, 'o-', linewidth=2, markersize=8, color='#1f77b4', label='User-based CF')
    axes[1, 0].plot([user_best_k_mae], [user_results[user_best_k_mae]['MAE']], 'r*', 
                     markersize=20, label=f'Best k={user_best_k_mae}')
    axes[1, 0].set_xlabel('Number of Neighbors (k)', fontsize=12)
    axes[1, 0].set_ylabel('MAE', fontsize=12)
    axes[1, 0].set_title('User-based CF: MAE vs k', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_xticks(k_values)
    
    axes[1, 1].plot(k_values, item_mae, 'o-', linewidth=2, markersize=8, color='#ff7f0e', label='Item-based CF')
    axes[1, 1].plot([item_best_k_mae], [item_results[item_best_k_mae]['MAE']], 'r*', 
                     markersize=20, label=f'Best k={item_best_k_mae}')
    axes[1, 1].set_xlabel('Number of Neighbors (k)', fontsize=12)
    axes[1, 1].set_ylabel('MAE', fontsize=12)
    axes[1, 1].set_title('Item-based CF: MAE vs k', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_xticks(k_values)
    
    plt.tight_layout()
    plt.savefig('task_2g_best_k.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'task_2g_best_k.png'")
    plt.show()
    
    # Conclusion
    print("\n" + "-"*80)
    print("CONCLUSION:")
    print("-"*80)
    if user_best_k == item_best_k:
        print(f"The best k is THE SAME for both methods: k = {user_best_k}")
        print("This suggests that both user-based and item-based CF benefit from")
        print("the same neighborhood size in this dataset.")
    else:
        print(f"The best k is DIFFERENT for each method:")
        print(f"  - User-based CF: k = {user_best_k}")
        print(f"  - Item-based CF: k = {item_best_k}")
        print("This suggests that user-based and item-based CF have different")
        print("optimal neighborhood sizes for this dataset.")
    print("-"*80)


def main():
    """
    Main function to run all tasks
    """
    print("="*80)
    print("MOVIE RECOMMENDER SYSTEM - TASK 2")
    print("="*80)
    
    # Load data
    data = load_data('ratings_small.csv')
    
    # Task 2c: Basic comparison with 5-fold CV
    results_2c = task_2c_basic_comparison(data)
    
    # Task 2d: Compare models and identify best
    task_2d_compare_models(results_2c)
    
    # Task 2e: Examine similarity metrics
    user_sim_results, item_sim_results = task_2e_similarity_metrics(data)
    
    # Task 2f: Examine neighbor impact
    user_neighbor_results, item_neighbor_results, k_values = task_2f_neighbor_impact(data)
    
    # Task 2g: Identify best k
    task_2g_best_k(user_neighbor_results, item_neighbor_results, k_values)
    
    print("\n" + "="*80)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("1. task_2d_model_comparison.png")
    print("2. task_2e_similarity_metrics.png")
    print("3. task_2f_neighbor_impact.png")
    print("4. task_2g_best_k.png")
    print("="*80)


if __name__ == "__main__":
    main()
