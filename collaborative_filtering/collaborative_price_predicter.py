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
from pyspark.sql import SparkSession
from pyspark.sql.functions import min, max

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

# -------------------------------------------------
# Normalize data between 0 and 1
# According to this formula:
# zi = (xi - min(x))/(max(x)-min(x))
# where 
# 	zi is the normalized value 
# 	xi is the original value prior to normalization
#	x is (x1, x2, ..., xi, ..., xn)
# -------------------------------------------------

# Obtain all the min and max of each column for normalization
maxCarat = dataFrame.select(max('carat')).collect()[0][0]
minCarat = dataFrame.select(min('carat')).collect()[0][0]

maxCut = dataFrame.select(max('cut')).collect()[0][0]
minCut = dataFrame.select(min('cut')).collect()[0][0]

maxColor = dataFrame.select(max('color')).collect()[0][0]
minColor = dataFrame.select(min('color')).collect()[0][0]

maxClarity = dataFrame.select(max('clarity')).collect()[0][0]
minClarity = dataFrame.select(min('clarity')).collect()[0][0]

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
