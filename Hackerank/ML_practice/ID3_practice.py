from ID3_Learning import *
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
iris_data = data['data']
iris_class = data['target']

label = ['sepal length', 'sepal width', 'petal length ', 'petal width']

dtree = ID3DTree()
dtree.loadDataSet2(data= np.insert(iris_data, 4, iris_class, axis=1), labels=label)
dtree.train()

#print(dtree.tree)
vector =['6.3', '2.5', '5.', '1.9']   #['6.3', '2.5', '5.', '1.9', '2']
print("Real: 2", "Predict:", dtree.predict(dtree.tree, label, vector))
