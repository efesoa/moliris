from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import operator
from numpy import dot

# Reading csv file and splitting the measurements from the specie name
# data.to_html() function converts the csv file (data) into html table format
data = pd.read_csv('iris.data', header=None,
                   names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'specie'])
x = data.iloc[:, 0:4]
y = data.iloc[:, 4]
X = np.array(x)
Y = np.array(y)
data_table = data.to_html()


# Creating functions for Euclidean distance, Cosine similarity, Manhattan distance
# and K-Nearest Neighbor (knn)
def euclidean_distance(a, b, length):
    distance = 0
    for x in range(length):
        distance += np.square(a[x] - b[x])

    return np.sqrt(distance)


def cosine_similarity(a, b):
    return dot(a,b) / ( (dot(a,a) **.5) * (dot(b,b) ** .5) )


def manhattan(a, b):
    return sum(abs(a - b))


def knn(dataset, obj, k):

    distances = {}
    sort = {}
    length = obj.shape[1]

    # Calculating euclidean distance between each row of dataset and object
    for x in range(len(dataset)):

        dist = euclidean_distance(obj, dataset.iloc[x], length)
        distances[x] = dist[0]

    # Sorting them on the basis of distance
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
    sorted_d1 = sorted(distances.items())

    neighbors = []

    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])
        counts = {"Iris-setosa":0,"Iris-versicolor":0,"Iris-virginica":0}

    # Calculating the most freq class in the neighbors
    for x in range(len(neighbors)):
        response = dataset.iloc[neighbors[x]][-1]

        if response in counts:
            counts[response] += 1
        else:
            counts[response] = 1

    sortedvotes = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    return(sortedvotes[0][0], neighbors)


# Instantiating app with Flask(__name__)
# Creating routes [home page, similarity page, similar objects page, and about page]
app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/similarity', methods=['GET', 'POST'])
def similarity():
    data_view = data_table
    if request.method == 'POST':
        obj1_sepal_length = float(request.form['fsl'])
        obj1_sepal_width = float(request.form['fsw'])
        obj1_petal_length = float(request.form['fpl'])
        obj1_petal_width = float(request.form['fpw'])
        obj2_sepal_length = float(request.form['ssl'])
        obj2_sepal_width = float(request.form['ssw'])
        obj2_petal_length = float(request.form['spl'])
        obj2_petal_width = float(request.form['spw'])
        obj = np.array([[obj1_sepal_length, obj1_sepal_width, obj1_petal_length, obj1_petal_width],
                                   [obj2_sepal_length, obj2_sepal_width, obj2_petal_length, obj2_petal_width]])
        obj1 = obj[0]
        obj1_d = pd.DataFrame(obj1)
        obj2 = obj[1]
        obj2_d = pd.DataFrame(obj2)
        predicting_object1, n = knn(data, obj1_d, 1)
        predicting_object2, n = knn(data, obj2_d, 1)
        cal_manhattan = manhattan(obj[0], obj[1])
        cal_euclidean = euclidean_distance(obj[0], obj[1], 4)
        cal_cosine = cosine_similarity(obj[0], obj[1])
        return render_template('similarity.html', predicting_first_object=predicting_object1,
                               predicting_second_object=predicting_object2,
                               obj=obj, manhattan=cal_manhattan, cosine=cal_cosine, euclidean=cal_euclidean,
                               obj1_sl=obj1_sepal_length, obj1_sw=obj1_sepal_width, obj1_pl=obj1_petal_length,
                               obj1_pw=obj1_petal_width, obj2_sl=obj2_sepal_length, obj2_sw=obj2_sepal_width,
                               obj2_pl=obj2_petal_length, obj2_pw=obj2_petal_width, data_view=data_view)
    return render_template('similarity.html', data_view=data_view)


@app.route('/similar_objects', methods=['GET', 'POST'])
def similar_objects():
    data_view = data_table
    if request.method == 'POST':
        sepal_length = float(request.form['sl'])
        sepal_width = float(request.form['sw'])
        petal_length = float(request.form['pl'])
        petal_width = float(request.form['pw'])
        k = int(request.form['k_n'])
        obj = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        obj1_d = pd.DataFrame(obj)
        predicting_object_knn, nn = knn(data, obj1_d, k)
        a = []
        b = []
        for n in nn:
            a.append(Y[n])
            b.append(X[n-1])

        return render_template('similar_objects.html', predicting_object_knn=predicting_object_knn,
                               sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length, k=k,
                               petal_width=petal_width, data_view=data_view, nearest_neighbors=nn, a=a, b=b, Y=Y)
    return render_template('similar_objects.html', data_view=data_view)


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
