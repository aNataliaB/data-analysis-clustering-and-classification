# Air Quality
An unsupervised machine learning project focused on the segmentation of atmospheric states using cluster analysis based on chemical sensor data.

## Project Overview
The primary goal of this research is to identify natural clusters corresponding to different air pollution profiles (e.g., traffic rush hours vs. nighttime conditions) using the "Air Quality" dataset from the UCI Machine Learning Repository.

### Key Features:
* **Data Preprocessing:** Handling missing sensor values (encoded as -200) through data cleaning and imputation, followed by feature normalization to ensure equal weighting in distance-based algorithms.
* **Exploratory Analysis:** Statistical evaluation (mean, standard deviation, skewness) and distribution analysis of 6 key diagnostic variables (CO, NMHC, C6H6, NOx, NO2, and Temperature).
* **Cluster Analysis:** Implementing and comparing two distinct clustering approaches:
  * **K-Means Clustering** (Non-hierarchical)
  * **Ward's Method** (Hierarchical agglomerative clustering based on variance analysis)
* **Cluster Evaluation:** Determining the optimal number of clusters ($k=3$) using the Elbow Method. Models were evaluated and compared using the **Silhouette Score** and the **Variance Reduction Variance Metric ($R^2$)**.

## Team & Collaboration
This project was developed in collaboration with a classmate.
* **My Contribution:** Participated in data pre-processing and cleaning routines, generated diagnostic visualizations (histograms and boxplots), implemented cluster analysis algorithms, performed PCA dimensional reduction for cluster visualization, and co-authored the final research report.

## Technologies & Methodologies
* **Language/Tools:** Python / Analytical software environment
* **Key Algorithms:** K-Means, Ward's Hierarchical Clustering, Principal Component Analysis (PCA)
* **Metrics Used:** Elbow Method, Silhouette Coefficient, Variance Decomposition

## Repository Structure
* `AirQualityUCI.csv` / `AirQualityUCI.xlsx` - Source datasets from the UCI repository.
* `Ocena_jakości_powietrza.pdf` - Comprehensive scientific and statistical report detailing findings and methodology (in Polish).
