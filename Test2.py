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

    ratings = ratingsRaw.filter(lambda line: line != ratingsHeader) \
        .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0], tokens[1], tokens[2])).cache()

    training_RDD, validation_RDD, test_RDD = ratings.randomSplit([6, 2, 2], seed=0)
    validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
    test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))


    sameModel = MatrixFactorizationModel.load(sc, "C:/Users/user/Desktop/alsmodel")
    predictions = sameModel.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())

    print('For testing data the RMSE is %s' % (error))
    sc.stop()