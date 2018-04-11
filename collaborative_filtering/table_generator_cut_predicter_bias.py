import subprocess

# ----------------------------------------------
# Run this script from the main project folder
# to generate the User Item RMSE table
# ----------------------------------------------

# Script arguments
cols = ["carat", "color", "clarity", "depth", "table", "price", "x", "y", "z"]
max_iter = str(10)
seed = str(123)
datafile = "data/diamonds.csv"

# Print table header
print("User,Item,RMSE,ErrorPercentage,RMSEBias")

# Try each candidate pair (carat, color), (carat, clarity), ... (z,y)
# and run the cut_predicter.py for each pair, making a table
for user in cols:
	for item in cols:
		if user == item:
			continue
		# Run script
		command = "python collaborative_filtering/cut_predicter_bias.py " + datafile + " " + seed + " " + user + " " + item + " " + max_iter
		process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		code = process.wait()
		rmse_errorPercentage = process.stdout.read().decode("utf-8")
		
		# Print the table
		print (user + "," + item + "," + rmse_errorPercentage)




