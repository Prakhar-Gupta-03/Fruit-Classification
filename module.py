import numpy as np
import csv
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def load_data(name, with_label):
    if(with_label):
        Feature = []
        Label = []
        with open(name, 'r') as f:
            reader = csv.reader(f)
            train_data = list(reader)
            # remove the first row
            train_data = train_data[1:]
            for row in train_data:
                Feature.append(np.array([float(x) for x in row[1:-1]]))
                Label.append(row[-1])  
        Features = np.array(Feature)
        Labels = np.array(Label)
        return Features, Labels
    else:
        Features = []
        with open (name, mode='r') as file:
            csvFile = csv.reader(file)
            data = list(csvFile)
            data = data[1:]
            for data_point in data:
                Features.append(np.array([float(x) for x in data_point[1:]]))
        Features = np.array(Features)
        return Features
    
def LOF(num_neighbors, Features, Labels):
    clf = LocalOutlierFactor(n_neighbors=num_neighbors)
    predicted = clf.fit_predict(Features, Labels)
    f, l = [], []
    for i in range(len(Features)):
        if(predicted[i]==1):
            f.append(Features[i]) 
            l.append(Labels[i])
    return f, l

def Pca(num_components, Features_Train, Features_Test):
    pca = PCA(n_components=num_components)
    Features_Train = pca.fit_transform(Features_Train)
    Features_Test = pca.transform(Features_Test)
    return Features_Train, Features_Test

def LDA(Features_Train, Labels_Train, Features_Test):
    lda = LinearDiscriminantAnalysis()
    Features_Train = lda.fit_transform(Features_Train, Labels_Train)
    Features_Test = lda.transform(Features_Test)
    return Features_Train, Features_Test

def Agglomerative(num_clusters, Features_Train, Features_Test):
    agglomerative = AgglomerativeClustering(n_clusters=num_clusters)

    Cluster_Label_Train = agglomerative.fit_predict(Features_Train)
    new_Features_Train = np.expand_dims(Cluster_Label_Train, axis=1)
    new_Features_Train = np.concatenate((Features_Train, new_Features_Train), axis=1)

    Cluster_Label_Test = agglomerative.fit_predict(Features_Test)
    new_Features_Test = np.expand_dims(Cluster_Label_Test, axis=1)
    new_Features_Test = np.concatenate((Features_Test, new_Features_Test), axis=1)

    return new_Features_Train, new_Features_Test

def Logistic(max_itr, Features_Train, Labels_Train, Features_Test):
    
    lr = LogisticRegression(max_iter=max_itr)
    lr.fit(Features_Train, Labels_Train)
    Predicted_Label_Test = lr.predict(Features_Test)
    
    return Predicted_Label_Test

def Cross_Validation_Score(max_itr, Features, Labels, n_splits):
    kf = KFold(n_splits=n_splits)
    accuracy = 0
    for i, (train_index, test_index) in enumerate(kf.split(Features)):

        Features_Train, Labels_Train, Features_Test, Labels_Test = [], [], [], []
        for i in train_index:
            Features_Train.append(Features[i])
            Labels_Train.append(Labels[i])
        
        for i in test_index:
            Features_Test.append(Features[i])
            Labels_Test.append(Labels[i])

        Features_Train, Features_Test = Pca(num_components=417, Features_Train=Features_Train, Features_Test=Features_Test)

        Features_Train, Features_Test = LDA(Features_Train, Labels_Train, Features_Test)

        Features_Train, Features_Test = Agglomerative(num_clusters=4, Features_Train=Features_Train, Features_Test=Features_Test)

        Labels_Test_Predicted = Logistic(max_itr=max_itr, Features_Train=Features_Train, Labels_Train=Labels_Train, Features_Test=Features_Test)

        accuracy += accuracy_score(Labels_Test, Labels_Test_Predicted)
        
    print("Cross Validation Score : ", accuracy/n_splits)

def write_csv(name, Labels):
    with open(name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Category'])
        for i in range(len(Labels)):
            writer.writerow([i, Labels[i]])