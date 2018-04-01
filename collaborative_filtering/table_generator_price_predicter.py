import subprocess

# ----------------------------------------------
# Run this script from the main project folder
# to generate the User Item RMSE table
# ----------------------------------------------

# Script arguments
cols = ["carat", "color", "clarity", "cut", "depth", "table", "x", "y", "z"]
max_iter = str(10)
seed = str(123)
datafile = "data/diamonds.csv"

# Print table header
print("User,Item,RMSE")

# Try each candidate pair (carat, color), (carat, clarity), ... (z,y)
# and run the cut_predicter.py for each pair, making a table
for user in cols:
	for item in cols:
		if user == item:
			continue
		# Run script
		command = "python collaborative_filtering/price_predicter.py " + datafile + " " + seed + " " + user + " " + item + " " + max_iter
		process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		code = process.wait()
		rmse = str(float(process.stdout.read().decode("utf-8")))
		
		# Print the table
		print (user + "," + item + "," + rmse)




