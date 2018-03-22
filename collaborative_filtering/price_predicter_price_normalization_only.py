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
from pyspark.sql.functions import min, max, col, when, desc
from pyspark.sql.types import DoubleType

spark = SparkSession.builder.getOrCreate()

# -------------------------------------------------
# Obtain the system arguments
# -------------------------------------------------
print('Reading arguments...')
dataFilePath = sys.argv[1]
seedValue = int(sys.argv[2])
userColumn = sys.argv[3]
itemColumn = sys.argv[4]
maxIteration = int(sys.argv[5])

# -------------------------------------------------
# Read the input file and cast the numeric column
# to double type
# -------------------------------------------------
print('Reading data file...')
dataFrame = spark.read.csv(dataFilePath, header=True)

print('Casting numeric data to Double...')
dataFrame = dataFrame.withColumn("caratDouble", dataFrame["carat"].cast(DoubleType()))
dataFrame = dataFrame.withColumn("depthDouble", dataFrame["depth"].cast(DoubleType()))
dataFrame = dataFrame.withColumn("tableDouble", dataFrame["table"].cast(DoubleType()))
dataFrame = dataFrame.withColumn("priceDouble", dataFrame["price"].cast(DoubleType()))
dataFrame = dataFrame.withColumn("xDouble", dataFrame["x"].cast(DoubleType()))
dataFrame = dataFrame.withColumn("yDouble", dataFrame["y"].cast(DoubleType()))
dataFrame = dataFrame.withColumn("zDouble", dataFrame["z"].cast(DoubleType()))

dataFrame = dataFrame.select("caratDouble","color", "clarity", "depthDouble", "tableDouble", "priceDouble", "xDouble", "yDouble", "zDouble", "cut") \
		.withColumnRenamed("caratDouble", "carat") \
		.withColumnRenamed("depthDouble", "depth") \
		.withColumnRenamed("tableDouble", "table") \
		.withColumnRenamed("priceDouble", "price") \
		.withColumnRenamed("xDouble", "x") \
		.withColumnRenamed("yDouble", "y") \
		.withColumnRenamed("zDouble", "z")
dataFrame.show()


# -------------------------------------------------
# Transform the labels to numbers 
# -------------------------------------------------
print('Mapping categorical data to numerical...')
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
dataFrame.show()


# -------------------------------------------------
# Normalize the price between 0 and 1
# According to this formula:
# zi = (xi - min(x))/(max(x)-min(x))
# where 
# 	zi is the normalized value 
# 	xi is the original value prior to normalization
#	x is (x1, x2, ..., xi, ..., xn)
# -------------------------------------------------

# Obtain all the min and max of each column for normalizatization
print('Obtaining maximums and minimums for normalization...')
minMax = dataFrame.agg(max('price'), min('price')).collect()[0]
maxPrice = float(minMax['max(price)'])
minPrice = float(minMax['min(price)'])


# Normalize the data
print('Normalizing data...')
scaleMax = 1000
scaleMin = 1
nPriceCol = ((col("price") - minPrice) / (maxPrice - minPrice)) * (scaleMax-scaleMin) + scaleMin
dataFrame = dataFrame.withColumn("nPrice", nPriceCol)
dataFrame = dataFrame.select("carat", "color", "clarity", "depth", "table", "nPrice", "x", "y", "z", "cut") \
		.withColumnRenamed("nPrice", "price")
dataFrame.show()

# -------------------------------------------------
# Create a training and test dataset
# and ALS model
# -------------------------------------------------
(training, test) = dataFrame.randomSplit([0.8, 0.2], seed=seedValue)
als = ALS(maxIter=maxIteration, regParam=0.01, userCol=userColumn, itemCol=itemColumn, ratingCol="price", coldStartStrategy="drop", rank=70)
als.setSeed(seedValue)
model = als.fit(training)

# -------------------------------------------------
# Evaluate the prediction model 
# -------------------------------------------------
predictions = model.transform(test)
predictions.show()

predictions.orderBy(desc("price")).show()

evaluator = RegressionEvaluator(metricName="rmse", labelCol="price", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(float(rmse))
