# -------------------------------------------------
# The following script takes in as parameter:
# (1) The file path of the diamond dataset
# (2) The seed value
# (3) The user column name in the dataset
# (4) The item column name in the dataset
# (5) Max iteration for the ALS model 
#
# It then splits the dataset into
# a training and a test set (80%, 20% respectively),
# and finally, it builds the ALS and trains the model
# using RMSE to evaluate the predictions.
#
# The final value of the RMSE is printed.
#
# No normalization is performed, only categorical
# data to numerical data mapping is done.
# -------------------------------------------------

import sys
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.sql.functions import min, max, col, when, desc, round
from pyspark.sql.types import DoubleType, IntegerType

spark = SparkSession.builder.getOrCreate()

# -------------------------------------------------
# Obtain the system arguments
# -------------------------------------------------
dataFilePath = sys.argv[1]
seedValue = int(sys.argv[2])
userColumn = sys.argv[3]
itemColumn = sys.argv[4]
maxIteration = int(sys.argv[5])

# -------------------------------------------------
# Read the input file and cast the numeric column
# to double type
# -------------------------------------------------
dataFrame = spark.read.csv(dataFilePath, header=True)

dataFrame = dataFrame.withColumn("caratDouble", dataFrame["carat"].cast(DoubleType()))
dataFrame = dataFrame.withColumn("depthDouble", dataFrame["depth"].cast(DoubleType()))
dataFrame = dataFrame.withColumn("tableDouble", dataFrame["table"].cast(DoubleType()))
dataFrame = dataFrame.withColumn("priceInt", dataFrame["price"].cast(IntegerType()))
dataFrame = dataFrame.withColumn("xDouble", dataFrame["x"].cast(DoubleType()))
dataFrame = dataFrame.withColumn("yDouble", dataFrame["y"].cast(DoubleType()))
dataFrame = dataFrame.withColumn("zDouble", dataFrame["z"].cast(DoubleType()))

dataFrame = dataFrame.select("caratDouble","color", "clarity", "depthDouble", "tableDouble", "priceInt", "xDouble", "yDouble", "zDouble", "cut") \
		.withColumnRenamed("caratDouble", "carat") \
		.withColumnRenamed("depthDouble", "depth") \
		.withColumnRenamed("tableDouble", "table") \
		.withColumnRenamed("priceInt", "price") \
		.withColumnRenamed("xDouble", "x") \
		.withColumnRenamed("yDouble", "y") \
		.withColumnRenamed("zDouble", "z")

# -------------------------------------------------
# Change the double to int, so that they can be 
# used as a user/item 
# -------------------------------------------------

# Carat: They have at most two decimals, so multiply
# by 100 (d.dd --> ddd)
dataFrame = dataFrame.withColumn("caratInt", round(dataFrame["carat"] * 100).cast(IntegerType()))

# Depth: They have at most one decimal, so multiply
# by 10 (d.d --> dd)
dataFrame = dataFrame.withColumn("depthInt", round(dataFrame["depth"] * 10).cast(IntegerType()))

# Table: They have at most one decimal, so multiply
# by 10 (d.d --> dd)
dataFrame = dataFrame.withColumn("tableInt", round(dataFrame["table"] * 10).cast(IntegerType()))

# x,y and z: They have at most two decimals, so multiply
# by 100 (d.dd --> dd)
dataFrame = dataFrame.withColumn("xInt", round(dataFrame["x"] * 100).cast(IntegerType()))
dataFrame = dataFrame.withColumn("yInt", round(dataFrame["y"] * 100).cast(IntegerType()))
dataFrame = dataFrame.withColumn("zInt", round(dataFrame["z"] * 100).cast(IntegerType()))

dataFrame = dataFrame.select("caratInt", "color", "clarity", "depthInt", "tableInt", "price", "xInt", "yInt", "zInt", "cut") \
		.withColumnRenamed("caratInt", "carat") \
		.withColumnRenamed("depthInt", "depth") \
		.withColumnRenamed("tableInt", "table") \
		.withColumnRenamed("xInt", "x") \
		.withColumnRenamed("yInt", "y") \
		.withColumnRenamed("zInt", "z")

# -------------------------------------------------
# Transform the labels to numbers 
# -------------------------------------------------
cutNumColumn = when(col("cut") == "Fair", 1) \
		.when(col("cut") == "Good", 2) \
		.when(col("cut") == "Very Good", 3) \
		.when(col("cut") == "Premium", 4) \
		.when(col("cut") == "Ideal", 5) \
		.otherwise(0)

colorNumColumn = when(col("color") == "J", 1) \
		.when(col("color") == "I", 2) \
		.when(col("color") == "H", 3) \
		.when(col("color") == "G", 4) \
		.when(col("color") == "F", 5) \
		.when(col("color") == "E", 6) \
		.when(col("color") == "D", 7) \
		.otherwise(0)

clarityNumColumn = when(col("clarity") == "I1", 1) \
		.when(col("clarity") == "SI2", 2) \
		.when(col("clarity") == "SI1", 3) \
		.when(col("clarity") == "VS2", 4) \
		.when(col("clarity") == "VS1", 5) \
		.when(col("clarity") == "VVS2", 6) \
		.when(col("clarity") == "VVS1", 7) \
		.when(col("clarity") == "IF", 8) \
		.otherwise(0)

dataFrame = dataFrame.withColumn("cutNum", cutNumColumn)
dataFrame = dataFrame.withColumn("colorNum", colorNumColumn)
dataFrame = dataFrame.withColumn("clarityNum", clarityNumColumn)

dataFrame = dataFrame.select("carat","colorNum", "clarityNum", "depth", "table", "price", "x", "y", "z", "cutNum") \
		.withColumnRenamed("colorNum", "color") \
		.withColumnRenamed("clarityNum", "clarity") \
		.withColumnRenamed("cutNum", "cut") 

# -------------------------------------------------
# Create a training and test dataset
# and ALS model
# -------------------------------------------------
(training, test) = dataFrame.randomSplit([0.8, 0.2], seed=seedValue)
als = ALS(maxIter=maxIteration, regParam=0.01, userCol=userColumn, itemCol=itemColumn, ratingCol="cut", coldStartStrategy="drop", rank=70)
als.setSeed(seedValue)
model = als.fit(training)

# -------------------------------------------------
# Evaluate the prediction model 
# -------------------------------------------------
predictions = model.transform(test)

evaluator = RegressionEvaluator(metricName="rmse", labelCol="cut", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(float(rmse))
