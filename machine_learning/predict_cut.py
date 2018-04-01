from pyspark.sql import Row, SparkSession
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf

spark = SparkSession.builder.master("local").appName("DiamondProject").getOrCreate()
sc = spark.sparkContext

# Load and parse the data file into an RDD of LabeledPoint.
data = MLUtils.loadLibSVMFile(sc, 'data/diamonds.data')
# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3],seed=123)

# Train a RandomForest model.
# Empty categoricalFeaturesInfo indicates all features are continuous.
# Note: Use larger numTrees in practice.
# Setting featureSubsetStrategy="auto" lets the algorithm choose.
model = RandomForest.trainClassifier(trainingData, numClasses=9, categoricalFeaturesInfo={},
                                     numTrees=25, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=20, maxBins=32, seed=123)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(testData.count())
result = testData.zip(predictions).collect()

#print('Learned classification forest model:')
#print(model.toDebugString())

with open('machine_learning/cut_predicted.txt', 'w') as f:
	for i in result:
		f.write(str(i))
		f.write('Test Error = ' + str(testErr))

labeled_result = labelsAndPredictions.map(lambda p: Row(cut=float(p[0]), predictions=float(p[1])))
result = spark.createDataFrame(labeled_result).show(25)

# Save and load model
model.save(sc, "machine_learning/models/DiamondClassificationModel")
sameModel = RandomForestModel.load(sc, "machine_learning/models/DiamondClassificationModel")
