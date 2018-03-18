# -------------------------------------------------
# The following script takes in as parameter:
# (1) The file path of the diamond dataset
# (2) The seed value
# (3) The user column name in the dataset
# (4) The iteam column name in the dataset
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
# Read the input file 
# -------------------------------------------------
dataFrame = spark.read.csv(dataFilePath, header=True)
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

# Transform the labels to numbers 
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
dataFrame.show()


# Obtain all the min and max of each column for normalizatization
maxCarat = dataFrame.select(max('carat')).collect()[0][0]
minCarat = dataFrame.select(min('carat')).collect()[0][0]

maxCut = dataFrame.select(max('cutNum')).collect()[0][0]
minCut = dataFrame.select(min('cutNum')).collect()[0][0]

maxColor = dataFrame.select(max('colorNum')).collect()[0][0]
minColor = dataFrame.select(min('colorNum')).collect()[0][0]

maxClarity = dataFrame.select(max('clarityNum')).collect()[0][0]
minClarity = dataFrame.select(min('clarityNum')).collect()[0][0]

maxDepth = dataFrame.select(max('depth')).collect()[0][0]
minDepth = dataFrame.select(min('depth')).collect()[0][0]

maxTable = dataFrame.select(max('table')).collect()[0][0]
minTable = dataFrame.select(min('table')).collect()[0][0]

maxPrice = dataFrame.select(max('price')).collect()[0][0]
minPrice = dataFrame.select(min('price')).collect()[0][0]

maxX = dataFrame.select(max('x')).collect()[0][0]
minX = dataFrame.select(min('x')).collect()[0][0]

maxY = dataFrame.select(max('y')).collect()[0][0]
minY = dataFrame.select(min('y')).collect()[0][0]

maxZ = dataFrame.select(max('z')).collect()[0][0]
minZ = dataFrame.select(min('z')).collect()[0][0]
	
print(maxCarat)

# Normalize the data
nCaratCol = (col("carat") - float(minCarat)) / (float(maxCarat) - float(minCarat))
nCutCol = (col("cutNum") - float(minCut)) / (float(maxCut) - float(minCut))
nClarityCol = (col("clarityNum") - float(minClarity)) / (float(maxClarity) - float(minClarity))
nDepthCol = (col("depth") - float(minDepth)) / (float(maxDepth) - float(minDepth))
nTableCol = (col("table") - float(minTable)) / (float(maxTable) - float(minTable)) 
nPriceCol = (col("price") - float(minPrice)) / (float(maxPrice) - float(minPrice))
nXCol = (col("x") - float(minX)) / (float(maxX) - float(minX))
nYCol = (col("y") - float(minY)) / (float(maxY) - float(minY))
nZCol = (col("z") - float(minZ)) / (float(maxZ) - float(minZ))

dataFrame = dataFrame.withColumn("nCarat", nCaratCol)
dataFrame = dataFrame.withColumn("nCut", nCutCol)
dataFrame = dataFrame.withColumn("nClarity", nClarityCol)
dataFrame = dataFrame.withColumn("nDepth", nDepthCol)
dataFrame = dataFrame.withColumn("nTable", nTableCol)
dataFrame = dataFrame.withColumn("nPrice", nPriceCol)
dataFrame = dataFrame.withColumn("nX", nXCol)
dataFrame = dataFrame.withColumn("nY", nYCol)
dataFrame = dataFrame.withColumn("nZ", nZCol)

dataFrame.show()

# -------------------------------------------------
# Create a training and test dataset
# and ALS model
# -------------------------------------------------
(training, test) = dataFrame.randomSplit([0.8, 0.2], seed=seedValue)
als = ALS(maxIter=maxIteration, regParam=0.01, userCol=userColumn, itemCol=itemColumn, rating="price", coldStartStrategy="drop", rank=70)
als.setSeed(seedValue)
model = als.fit(training)

# -------------------------------------------------
# Evaluate the prediction model 
# -------------------------------------------------
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(float(rmse))
