import numpy as np

# Open the file in read mode ('r')
with open('./ML_Assignment_2_2024_data/traindata.txt', 'r') as file:
    # Read all lines from the file
    data = file.readlines()

for i in range(len(data)):
    parts = data[i].split(',')
    data[i] = [float(part) for part in parts]
    data[i] = np.array(data[i])

with open('./ML_Assignment_2_2024_data/trainlabels.txt', 'r') as file:
    labels = [line.strip() for line in file]

for i in range(len(labels)):
    parts = labels[i].split('\n')
    labels[i] = [int(part) for part in parts if part]
labels = [int(item) for sublist in labels for item in sublist]

print(labels)