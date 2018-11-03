import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import preprocessing
import argparse
import graphviz

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)
ap.add_argument("-attri", "--attributes", type=str, required=True)
ap.add_argument("-cat", "--category", type=str)
ap.add_argument("-c", "--class", type=int, required=True)
args = vars(ap.parse_args())
le = preprocessing.LabelEncoder()

balance_data = pd.read_csv(args['dataset'], sep=',')

X = balance_data.values[:, int(args['attributes'].split(':')[0]):int(args['attributes'].split(':')[1])]
Y = balance_data.values[:, args['class']]
Y = Y.astype('int')

if args['category']:
    for i in range(int(args['category'].split(':')[0]), int(args['category'].split(':')[1])):
        col = int(args['category'].split(':')[0])
        le.fit(list(set(X[:, i - col])))
        X[:, i - col] = le.transform(X[:, i - col])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
clf = tree.DecisionTreeClassifier(random_state=10)
clf.fit(X_train, y_train)
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("graph")
acc = clf.score(X_test, y_test)
print("accuracy: " + str(acc))
