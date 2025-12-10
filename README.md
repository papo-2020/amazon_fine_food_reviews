# Amazon Fine Food Reviews: Unsupervised Learning & Collaborative Filtering

> Exploring dimensionality reduction, clustering, and collaborative filtering techniques on a large-scale recommender system dataset

**Course:** Unsupervised Learning (MS CS)  
**Institution:** University of Colorado Boulder  
**Author:** Philipp Adrian Pohlmann

## Overview

This project applies unsupervised learning techniques from the course to the Amazon Fine Food Reviews dataset, treating it as a large, sparse user-item utility matrix. The goal is to understand how different collaborative filtering approaches perform on realistic, sparse data and to compare memory-based vs model-based methods.

### Methods Applied:
- **Dimensionality Reduction** (PCA/TruncatedSVD)
- **Clustering** (K-means on item embeddings)
- **Memory-Based Collaborative Filtering** (item-item cosine similarity)
- **Model-Based Collaborative Filtering** (matrix factorization with SGD)

## Dataset

- **Source:** Amazon Fine Food Reviews (Kaggle)
- **Link:** https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
- **Size:** 568,454 reviews
- **Users:** 256,059 unique reviewers
- **Items:** 74,258 unique products
- **Ratings:** 1-5 stars (heavily skewed to 5 stars - 64% of reviews)
- **Split:** 511,608 training / 56,846 validation (90/10 split)

### Characteristics:
- Extremely sparse matrix (most users rate very few items)
- Long-tail distribution (most items have very few ratings)
- Only 1,632 items (2%) have ≥50 ratings
- Rating distribution: 5★: 363K | 4★: 81K | 3★: 43K | 2★: 30K | 1★: 52K

## 📈 Results Summary

| Model | RMSE (val) | MAE (val) | Notes |
|-------|-----------|----------|-------|
| Global Mean Baseline | 1.90 | 0.80 | No personalization |
| Item-Based CF (k=20) | 1.80 | 0.58 | Local neighborhood, popular items only |
| **Matrix Factorization** | **1.16** | 0.89 | **35% improvement in RMSE** |

### Key Results:
- **Matrix Factorization wins** with RMSE 1.16 (35% better than item-based CF)
- Captures global latent structure with 30 factors
- Item-based CF performs well on MAE using local neighborhoods
- Global mean baseline surprisingly decent due to skewed ratings

## In-depth EDA

EDA revealed:
- Extreme sparsity in user-item matrix
- Rating distribution heavily skewed to positive reviews (64% are 5-star)
- Long-tail distribution of user activity and item popularity
- Only 2% of items have ≥50 ratings (1,632 of 74,258)

## Models Implemented

### 1. Dimensionality Reduction (PCA/TruncatedSVD)
- Applied to item rating vectors to visualize latent space
- **Finding:** Only 3.2% variance in first component
- Variance spread across many dimensions (typical for sparse recommenders)

### 2. K-Means Clustering
- Clustered item embeddings in reduced space
- Selected K=10 after grid search
- Discovered natural item groupings based on rating patterns

### 3. Item-Based Collaborative Filtering
- Memory-based approach using item-item cosine similarity
- k=20 nearest neighbors for prediction
- Restricted to items with ≥50 ratings for computational efficiency
- **Result:** RMSE 1.80, MAE 0.58

### 4. Matrix Factorization (SGD)
- Learns 30-dimensional latent factors for users and items
- Includes user and item biases
- L2 regularization (λ=0.02) to prevent overfitting
- Trained with SGD over 5 epochs (lr=0.01)
- **Result:** RMSE 1.16, MAE 0.89 ✓

## Findings

### 1. Extreme Sparsity Requires Global Models
With only 568K ratings across 256K users and 74K items, most entries in the utility matrix are empty. PCA analysis confirms variance is spread across many dimensions (only 3.2% in PC1). Matrix factorization handles this by learning global latent structure.

### 2. Matrix Factorization Captures Global Structure Best
MF achieves 35% lower RMSE than item-based CF by compressing the high-dimensional matrix into 30 latent factors. This allows it to generalize across many more items than local neighborhood methods.

### 3. Skewed Ratings Affect All Models
With 64% of reviews being 5-star, even the global mean baseline achieves reasonable MAE (0.80). However, it completely ignores personalization and user-item structure.

### 4. Trade-offs Matter in Practice
While MF achieves best RMSE, it requires:
- More complex training procedure (SGD, multiple epochs)
- Careful hyperparameter tuning (lr, reg, factors, epochs)
- More computational resources

Item-based CF is simpler to implement and update incrementally with new data.

## Next Steps

With more time, the following extensions would be valuable:

1. **Hyperparameter Tuning:**
   - Grid search over number of latent factors (10, 20, 30, 50)
   - Optimize regularization strength
   - Experiment with learning rate schedules

2. **Incorporate Review Text:**
   - Use text features for cold-start items
   - Combine collaborative and content-based filtering
   - Initialize item embeddings from text representations

3. **Advanced Evaluation:**
   - Ranking metrics (NDCG, hit rate@k)
   - A/B testing framework
   - Coverage and diversity metrics

4. **Temporal Effects:**
   - Model changing user preferences over time
   - Incorporate review timestamp information

5. **Deep Learning Extensions:**
   - Neural collaborative filtering
   - Autoencoders for implicit feedback
   - Attention mechanisms for reviews

## References

- Dataset: [Amazon Fine Food Reviews on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30-37.
