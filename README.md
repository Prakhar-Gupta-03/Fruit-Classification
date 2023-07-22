# SML_Project

## Project Description
This project was done as a part of the (Kaggle competition)[https://www.kaggle.com/competitions/sml-project/] for the project as a part of the course Statistical Machine Learning (CSE342) at IIIT Delhi. The project was done by a team of two members - (Prakhar Gupta)[https://github.com/Prakhar-Gupta-03] and (Shreyas Kabra)[https://github.com/shreyas21563]. 
The project provided a dataset for fruit classification, with 1216 data points and we had to classify the data into one out of 20 labels. 

## Approach
We tried to implement several techniques to improve the accuracy of our model. Since the model had more parameters than the number of data points, we tried to use principal component analysis (PCA) and linear discriminant analysis (LDA) to reduce the number of dimensions. We tried to experiment with random forests, neural networks, and several unsupervised clustering algorithms as well- k-means, DBSCAN, Agglomerative clustering, etc. Then, we tried to work with several ensembling methods to make the model more robust. 
We have added a detailed report of our approach in the file `report.pdf` in the repository, which also follows an IEEE format for the report. The source code for the final submission can be found in the file `source_code.ipynb`. We achieved an accuracy of 85.990% on the training dataset (public leaderboard) and 81.730% on the testing dataset (private leaderboard), which hints at a slight overfitting of the model.