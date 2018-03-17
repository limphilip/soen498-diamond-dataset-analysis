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
dataFrame = spark.read.csv(dataFilePath, header=False)

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
minMax = dataFrame.select(max('carat'), min('carat'), \
	max('cut'), min('cut'), \
	max('color'), min('color'), \
	max('clarity'), min('clarity'), \
	max('depth'), min('depth'), \
	max('table'), min('table'), \
	max('price'), min('price'), \
	max('x'), min('x'), \
	max('y'), min('y'), \
	max('z'), min('z')).collect()
	
print(minMax)


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
