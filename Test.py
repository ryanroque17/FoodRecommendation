import os

os.environ["SPARK_HOME"] = "C:/Users/user/AppData/Local/Programs/Python/Python35/Lib/site-packages/pyspark"
import math

from pyspark import SparkContext
from pyspark.sql import SparkSession
#from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, CrossValidator
from pyspark.sql.types import DoubleType, StringType, StructField, StructType, IntegerType
if __name__ == "__main__":
    sc = SparkContext(appName="PythonWordCount")
    ratingsRaw = sc.textFile("rating_final.csv", 1)
    # rdd = sc.parallelize([("a", 1)])
    # print(hasattr(rdd, "select"))

    ratingsHeader = ratingsRaw.take(1)[0]

    # spark = SparkSession(sc).builder.getOrCreate()
    # print(hasattr(rdd, "select"))

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
    ranks = [4, 8, 12]
    regularization_parameters = [0.1, 0.11]
    iterations = [18, 19]
    errors = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    err = 0
    tolerance = 0.02

    min_error = float('inf')
    best_rank = -1
    best_iteration = -1
    best_regParam = -1

    for rank in ranks:
        for regParam in regularization_parameters:
            for iteration in iterations:
                model = ALS.train(training_RDD, rank, seed=seed, iterations=iteration,
                                  lambda_=regParam)
                predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
                ratingsTuple = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2])))
                rates_and_preds = predictions.join(ratingsTuple).map(lambda tup: tup[1])
                # rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
                metrics = RegressionMetrics(rates_and_preds)
                error = metrics.rootMeanSquaredError
                mae = metrics.meanAbsoluteError
                # rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
                # error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())
                # mae = math.sqrt(rates_and_preds.map(lambda r: (abs(r[1][0] - r[1][1]))).mean())
                errors[err] = error
                err += 1
                print('For rank %s, iteration %s, param %s the RMSE and MAE are %s and %s' % (rank, iteration, regParam, error, mae))

                if error < min_error:
                    min_error = error
                    best_rank = rank
                    best_iteration = iteration
                    best_regParam = regParam

    print('The best model was trained with rank %s, param %s, iteration %s' % (best_rank, best_regParam, best_iteration))

    model = ALS.train(training_RDD, best_rank, seed=seed, iterations=best_iteration,
                      lambda_=best_regParam)
    predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0],r[1]), r[2]))
    ratingsTuple = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2])))
    rates_and_preds = predictions.join(ratingsTuple).map(lambda tup: tup[1])
    #rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    metrics = RegressionMetrics(rates_and_preds)


    #rmse = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())
#    mae = math.sqrt(rates_and_preds.map(lambda r: (abs(r[1][0] - r[1][1]))).mean())

    print(test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).take(3))
    print(predictions.take(3))
    #    print(predictions.takeOrdered(10, key=lambda x: -x[1]))


    print('For testing data the RMSE and MAE are %s and %s' % (metrics.rootMeanSquaredError, metrics.meanAbsoluteError))


    # best_rank = 10
    # (training, test) = ratings.randomSplit([0.8, 0.2], seed=0)
    # als = ALS(userCol="userID", itemCol="placeID", ratingCol="rating", coldStartStrategy="drop", nonnegative= True)
    # param_grid = ParamGridBuilder()\
    #     .addGrid(als.rank, [10])\
    #     .addGrid(als.maxIter, [10])\
    #     .addGrid(als.regParam, [.1]).build()
    # evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    # tvs = TrainValidationSplit(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator)
    # schema = StructType().add(StructField("userID", IntegerType(), True)).add(StructField("placeID", IntegerType(), True)).add(StructField("rating", DoubleType(), True))
    # dataframe = spark.createDataFrame(training, schema)
    # model = tvs.fit(dataframe)
    # best_model = model.bestModel
    # testDataFrame = spark.createDataFrame(test, schema)
    # predictions = best_model.transform(testDataFrame)
    # rmse = evaluator.evaluate(predictions)
    #
    # print("RMSE %s" %rmse)
    # evaluator.setMetricName("mae")
    # mae = evaluator.evaluate(predictions)
    # print("MAE %s" %mae)
    #
    # print("rank %s" %best_model.rank)
    # print("maxiter %s" %best_model._java_obj.parent().getMaxIter())
    # print("regparam %s" %best_model._java_obj.parent().getRegIter())




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