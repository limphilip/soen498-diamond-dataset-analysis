import math
from pyspark.sql import Row, SparkSession
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf

spark = SparkSession.builder.master("local").appName("DiamondProject").getOrCreate()
sc = spark.sparkContext

# Load and parse the data file into an RDD of LabeledPoint.
data = MLUtils.loadLibSVMFile(sc, 'data/diamonds_price.data')

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3],seed=123)

# Train a RandomForest model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
#  Note: Use larger numTrees in practice.
#  Setting featureSubsetStrategy="auto" lets the algorithm choose.
model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                     numTrees=25, featureSubsetStrategy="auto",
                                     impurity='variance', maxDepth=20, maxBins=32, seed=123)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)

testRMSE = math.sqrt(labelsAndPredictions.map(lambda lp: (lp[0] - lp[1])**2).sum() / float(testData.count()))

result = testData.zip(predictions).collect()

# Print the predictions to output file
with open('machine_learning/results/predicted_price.txt', 'w') as f:
        for i in result:
                f.write(str(i)+"\n")
        f.write('Test Root Mean Square Error = ' + str(testRMSE))

# Print the learned classication forest model to output file
with open('machine_learning/models/forest_model_predicted_price.txt', 'w') as f:
        f.write(model.toDebugString())

labeled_result = labelsAndPredictions.map(lambda p: Row(price=float(p[0]), predictions=float(p[1])))
result = spark.createDataFrame(labeled_result).show(25)
print('Test Root Mean Square Error = ' + str(testRMSE))
