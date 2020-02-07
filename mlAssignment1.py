import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
from sklearn import svm


def main():
    df = pd.read_csv('C:/Users/Robert/Documents/mlAssignment1Dataset.csv', header=0, delimiter=";")

    # y is the feature data, found in column 12 of the csv
    y = df.Column12

    # Get the numeric data columns
    x = df._get_numeric_data()

    # Call the cross validation function once for each classifier
    cross_val(neighbors.KNeighborsClassifier(n_neighbors=9), "K Nearest Neighbour", x, y)
    cross_val(svm.SVC(kernel='linear', C=1.0), "Linear SVC", x, y)


def cross_val(model, name, x, y):
    result = cross_val_score(model, x, y, cv=10)
    print(name + " Results: " + str(result) + "\n")
    print("Mean: {:0.3}\nStandard Deviation: +/- {:0.3}\n".format(result.mean(), result.std()))


main()
