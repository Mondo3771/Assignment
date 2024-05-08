# Open the file in read mode ('r')
with open('./ML_Assignment_2_2024_data/traindata.txt', 'r') as file:
    # Read all lines from the file
    data = file.readlines()

# Now, 'data' is a list where each element is a line from the file
print(len(data))

with open('./ML_Assignment_2_2024_data/trainlabels.txt', 'r') as file:
    # Read all lines from the file
    labels = file.readlines()
print(len(data[0]))
print(len(labels))

print(labels[])