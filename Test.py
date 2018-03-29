import os

os.environ["SPARK_HOME"] = "C:/Users/user/AppData/Local/Programs/Python/Python35/Lib/site-packages/pyspark"
from operator import add

from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
import math
from pyspark.mllib.recommendation import MatrixFactorizationModel


if __name__ == "__main__":
    sc = SparkContext(appName="PythonWordCount")
    ratingsRaw = sc.textFile("rating_final.csv", 1)
    ratingsHeader = ratingsRaw.take(1)[0]

    # moviesRaw = sc.textFile("geoplaces2.csv", 1)
    # moviesHeader = moviesRaw.take(1)[0]
    #
    # movies = moviesRaw.filter(lambda line: line != moviesHeader) \
    #     .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]), tokens[4])).cache()
    #
    # movieTitles = movies.map(lambda x: (int(x[0]), x[1]))
    #
    # print(movies.take(3))
    # print(movieTitles.take(3))

    ratings = ratingsRaw.filter(lambda line: line != ratingsHeader) \
        .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0], tokens[1], tokens[2])).cache()


    training_RDD, validation_RDD, test_RDD = ratings.randomSplit([6, 2, 2], seed=0)
    validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
    test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

    seed = 5
    iterations = 10
    regularization_parameter = 0.1
    ranks = [4, 8, 12]
    errors = [0, 0, 0]
    err = 0
    tolerance = 0.02

    min_error = float('inf')
    best_rank = -1
    best_iteration = -1
    for rank in ranks:
        model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                          lambda_=regularization_parameter)
        predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
        rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
        error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())
        errors[err] = error
        err += 1
        print('For rank %s the RMSE is %s' % (rank, error))
        if error < min_error:
            min_error = error
            best_rank = rank

    print('The best model was trained with rank %s' % best_rank)

    model = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
    predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0],r[1]), r[2]))
    rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())

    print('For testing data the RMSE is %s' % (error))

    # predictionsWithTitle = predictions.join(movieTitles)
    # print(predictions.take(3))
    # print(movieTitles.take(3))
    # print(predictionsWithTitle.take(10))
    # data = predictions.collect()
    # print(model)
    # model.save(sc, "C:/Users/user/Desktop/alsmodel")
    #=
    # sameModel = MatrixFactorizationModel.load(sc, "C:/Users/user/Desktop/alsmodel")
    # predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    # rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    # error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())
    #
    # print('For testing data the RMSE is %s' % (error))
    sc.stop()