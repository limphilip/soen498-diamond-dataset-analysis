# -------------------------------------------------
# The following script takes in as parameter:
# (1) The file path of the diamond dataset
# (2) The seed value
# (3) The user column name in the dataset
# (4) The item column name in the dataset
# (5) Max iteration for the ALS model 
#
# It then normalizes the data, split the dataset into
# a training and a test set (80%, 20% respectively),
# and finally, it builds the ALS and trains the model
# using RMSE to evaluate the predictions.
#
# The final value of the RMSE is printed.
# -------------------------------------------------

import sys
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.sql.functions import min, max, col, when
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
# Normalize data between 0 and 1
# According to this formula:
# zi = (xi - min(x))/(max(x)-min(x))
# where 
# 	zi is the normalized value 
# 	xi is the original value prior to normalization
#	x is (x1, x2, ..., xi, ..., xn)
# -------------------------------------------------
# Obtain all the min and max of each column for normalizatization
print('Obtaining maximums and minimums for normalization...')
minMax = dataFrame.agg(max('carat'), min('carat'), \
			max('cut'), min('cut'), \
			max('color'), min('color'), \
			max('clarity'), min('clarity'), \
			max('depth'), min('depth'), \
			max('table'), min('table'), \
			max('price'), min('price'), \
			max('x'), min('x'), \
			max('y'), min('y'), \
			max('z'), min('z')).collect()[0]
maxCarat = float(minMax['max(carat)'])
minCarat = float(minMax['min(carat)'])
maxCut = float(minMax['max(cut)'])
minCut = float(minMax['min(cut)'])
maxColor = float(minMax['max(color)'])
minColor = float(minMax['min(color)'])
maxClarity = float(minMax['max(clarity)'])
minClarity = float(minMax['min(clarity)'])
maxDepth = float(minMax['max(depth)'])
minDepth = float(minMax['min(depth)'])
maxTable = float(minMax['max(table)'])
minTable = float(minMax['min(table)'])
maxPrice = float(minMax['max(price)'])
minPrice = float(minMax['min(price)'])
maxX = float(minMax['max(x)'])
minX = float(minMax['min(x)'])
maxY = float(minMax['max(y)'])
minY = float(minMax['min(y)'])
maxZ = float(minMax['max(z)'])
minZ = float(minMax['min(z)'])


# Normalize the data
print('Normalizing data...')
nCaratCol = (col("carat") - minCarat) / (maxCarat - minCarat)
nCutCol = (col("cut") - minCut) / (maxCut - minCut)
nColorCol = (col("color") - minColor) / (maxColor - minColor)
nClarityCol = (col("clarity") - minClarity) / (maxClarity - minClarity)
nDepthCol = (col("depth") - minDepth) / (maxDepth - minDepth)
nTableCol = (col("table") - minTable) / (maxTable - minTable) 
nPriceCol = (col("price") - minPrice) / (maxPrice - minPrice)
nXCol = (col("x") - minX) / (maxX - minX)
nYCol = (col("y") - minY) / (maxY - minY)
nZCol = (col("z") - minZ) / (maxZ - minZ)

dataFrame = dataFrame.withColumn("nCarat", nCaratCol)
dataFrame = dataFrame.withColumn("nCut", nCutCol)
dataFrame = dataFrame.withColumn("nColor", nColorCol)
dataFrame = dataFrame.withColumn("nClarity", nClarityCol)
dataFrame = dataFrame.withColumn("nDepth", nDepthCol)
dataFrame = dataFrame.withColumn("nTable", nTableCol)
dataFrame = dataFrame.withColumn("nPrice", nPriceCol)
dataFrame = dataFrame.withColumn("nX", nXCol)
dataFrame = dataFrame.withColumn("nY", nYCol)
dataFrame = dataFrame.withColumn("nZ", nZCol)

dataFrame = dataFrame.select("nCarat", "nColor", "nClarity", "nDepth", "nTable", "nPrice", "nX", "nY", "nZ", "nCut") \
		.withColumnRenamed("nCarat", "carat") \
		.withColumnRenamed("nColor", "color") \
		.withColumnRenamed("nClarity", "clarity") \
		.withColumnRenamed("nDepth", "depth") \
		.withColumnRenamed("nTable", "table") \
		.withColumnRenamed("nPrice", "price") \
		.withColumnRenamed("nX", "x") \
		.withColumnRenamed("nY", "y") \
		.withColumnRenamed("nZ", "z") \
		.withColumnRenamed("nCut", "cut") 
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
evaluator = RegressionEvaluator(metricName="rmse", labelCol="price", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(float(rmse))
