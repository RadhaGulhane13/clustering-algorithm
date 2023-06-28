# Clustering Algorithm

## Objectives:
Objective is to perform clustering on three datasets. Choose suitable clustering algorithms, evaluate them on the datasets, and compare their performance.

The objectives of this assignment are:

- Understand how to select and evaluate suitable off-the-shelf clustering algorithms based on the characteristics of a dataset and the outcomes you need.
- Understand how to tune and evaluate a clustering algorithm to achieve good performance.

## Datasets:
- The file small_Xydf.csv is a two-dimensional dataset with 200 records. It contains columns X0, X1, and y. The y column is the actual cluster number that was produced by the dataset generation algorithm. Do not use it for the clustering algorithm. It will be used to evaluate your clustering algorithm below.

- The file large1_Xydf.csv is a two-dimensional dataset with 3000 records. It contains columns X0, X1, and y. The y column is the actual cluster number that was produced by the dataset generation algorithm. Do not use it for the clustering algorithm. It will be used to evaluate your clustering algorithm below.

- The file large2_Xydf.csv is a two-dimensional dataset with 3000 records. It contains columns X0, X1, and y. The y column is the actual cluster number that was produced by the dataset generation algorithm. Do not use it for the clustering algorithm. It will be used to evaluate your clustering algorithm below.

## Evaluation

## Evaluate the K-Means Algorithm

```
kmeans = KMeans(n_clusters=k, max_iter=1000, copy_x=True).fit(small_data[['X0','X1']])
t1_stop = process_time()
pred_y = kmeans.labels_
true_y = small_data['y']
pred_y_maped = remap_cluster_labels(true_y, pred_y)
pred_y_small["kmeans_clusters_"+str(k)+"_small"] = pred_y_maped
print_evaluation_results("kmeans_clusters_"+str(k)+"_small", small_data[['X0','X1']], true_y, pred_y_maped, t1_stop-t1_start)
wss = kmeans.inertia_
``` 

### Small Dataset
![K-Means clustering on small dataset](https://github.com/RadhaGulhane13/clustering-algorithm/blob/main/img/kmeans_small.png)

Based on the "kmeans cluster size" graph, it can be observed that the elbow point is significant in choosing the cluster value as 3 for the K-means clustering algorithm. Moreover, the K-means model with a cluster size of 3 is giving a significantly better accuracy of 97% compared to the other two K-means models with cluster sizes of 2 and 4. Thus, I decided to prefer cluster value as 3 for small dataset.

### Large1 Dataset
![K-Means clustering on large1 dataset](https://github.com/RadhaGulhane13/clustering-algorithm/blob/main/img/kmeans_large1.png)
- From the above analysis of the classification report and other performance metrics, it can be observed that K = 8 provides a good score for clusters. The within sum of squared errors (SSE) graph also shows that the elbow occurs at k=8, and it provides an accuracy of 87%.
- However, it is worth noting that the running time of the K-means model with k=8 is higher than the running time of the K-means models with k=10. Therefore, if training time is a concern, the K-means model with k=10 could be an optional choice.

### Large2 Dataset
![K-Means clustering on large2 dataset](https://github.com/RadhaGulhane13/clustering-algorithm/blob/main/img/kmeans_large2.png)
- Above within sum of squared errors (SSE) graph also shows that the elbow occurs at k=3, however, it provides an accuracy of 56%. While comparing k values 2, 3, and 4, we can observ the good performance for k = 2 with accuracy of 75%.
- Moreover, running time of kmeans model with k = 2 is significantly less than kmeasn model with cluster size 3 and 4.
In addition, the scatterplot visualization also supports our findings, as it shows that most of the predicted data points are able to match with the actual values, indicating a good level of accuracy in the clustering results.

## Evaluate the BIRCH clustering model 
``` 
brc_model1 = Birch(branching_factor=50, n_clusters=None, threshold=1.5)
brc_model1.fit(large2_data[['X0','X1']])
t1_stop = process_time()
birch_model1_time = t1_stop - t1_start
pred_y_birch_model1 = brc_model1.predict(large2_data[['X0','X1']])
remap_pred_y_birch_model1 = remap_cluster_labels(true_y_large2, pred_y_birch_model1)
```

### Observations:

- From the analysis of performance metrics and graphs, it can be observed that the BIRCH clustering model with a threshold of 1 performs better than the BIRCH model with a threshold of 1.5.
- The BIRCH model with a threshold of 1.5 shows an accuracy of 50%, whereas the BIRCH model with a threshold of 1 shows an accuracy of 75%. Additionally, other performance metrics such as recall and precision also show lower values for the first model.
- I also observe that, BIRCH with threshold value less or more than 1, shows performance degradation.
- Even though BIRCH model with threshold 1 doesn't give great performance, but it is performing better as compare with theshold valuse as more than or less than 1.

## Evaluate the Spectral Clustering model
```
t1_start = process_time()
sc_model1 = SpectralClustering(n_clusters=2, affinity='rbf', eigen_solver='arpack', gamma = 7)
sc_model1.fit(X_large2)
t1_stop = process_time()
pred_y_sc_model1 = sc_model1.labels_
pred_y_sc_maped_model1 = remap_cluster_labels(true_y_large2, pred_y_sc_model1)
```
### Observations:
- Spectral clustering model showed excellent performance for both gamma values of 7 and 9, achieving high accuracy, recall, and precision of 97%, 97%, and 96% respectively.
- Additionally, the scatter plots indicated that both models were able to correctly classify most of the data points, with only a few misclassifications in the Spectral clustering model with gamma value of 9.

## Compare Performance

- Spectral clustering model showed significant performance over Kmeans and BIRCH clustering models in terms of overall accuracy when applied to a large2 dataset.
- Kmeans clustering model showed good performance with small and large1 datasets where clusters were well distributed, but did not perform well with the large2 dataset where the class distribution showed unique patterns.
- Although Spectral clustering model performed well in terms of accuracy, recall, precision, etc., it required significantly more training time compared to Kmeans and BIRCH clustering models.

### Characteristics of the data might impact the clustering algorithms:

- Data distribution, dimensionality, noise and outliers, scale and units, missing values, cluster shape and size, cluster density and overlapping, and class imbalance are characteristics of the data that can impact clustering algorithms' performance.
- Irregular cluster shapes, varying cluster sizes, overlapping clusters, imbalanced data, and other similar characteristics may adversely affect clustering results and require careful consideration or preprocessing.

## Conclusion
- I will choose Spectral Clustring model as it gave significant perfomance with accuracy of 97% for large 2 dataset where the class distribution showed specific pattern. Spectral clustering is proven effective in handling datasets with non-linear and complex structures, which may have resulted in improved performance.
- However, it's worth considering that Spectral clustering may have taken more training time compared to Kmeans. If training time is a concern, Kmeans clustering algorithm could also be a viable choice, as it has shown good performance for both small and large1 datasets in your analysis. It's important to weigh the trade-offs between performance and training time when choosing the best clustering algorithm for your specific needs.