import numpy as np

# Open the file in read mode ('r')
with open('./ML_Assignment_2_2024_data/traindata.txt', 'r') as file:
    # Read all lines from the file
    data = file.readlines()

# Now, 'data' is a list where each element is a line from the file
# Assuming 'data' is your list of lines
for i in range(len(data)):
    # Split the line into parts by comma
    parts = data[i].split(',')
    # Convert each part to a float and replace the original line with the list of floats
    data[i] = [float(part) for part in parts]
    data[i] = np.array(data[i])

# Now, 'data' is a list of lists of float
with open('./ML_Assignment_2_2024_data/trainlabels.txt', 'r') as file:
    # Read all lines from the file  
    labels = [line.strip() for line in file]

for i in range(len(labels)):
    # Split the line into parts by newline
    parts = labels[i].split('\n')
    # Convert each part to an integer and replace the original line with the list of integers
    labels[i] = [int(part) for part in parts if part]

print(parts)


