import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # or RandomForestRegressor
from sklearn.metrics import accuracy_score  # or another appropriate metric
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier # If using an older version of scikit-learn
import joblib  # For newer versions

# Load training data
with open('traindata.txt', 'r') as file:
    data = file.readlines()
data = np.array([np.array([float(value) for value in line.split(',')]) for line in data])
df = pd.DataFrame(data)
# Load test data
with open('testdata.txt', 'r') as file:
    test_data = file.readlines()
test_data = np.array([np.array([float(value) for value in line.split(',')]) for line in test_data])

# Load training labels
with open('trainlabels.txt', 'r') as file:
    labels = [int(line.strip()) for line in file]

# print(data) 
# print(labels)
df['label'] = labels
# Optionally, assign generic feature names
# df.columns = [f'F_{i}' for i in range(df.shape[1])]
# Identify columns with negative numbers
negative_features = [col for col in df.columns if (df[col] < 0).any()]

# Drop these columns from df
df = df.drop(columns=negative_features)
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Ensure pandas is imported
import scipy.ndimage as ndimage

# Assuming df is already loaded and the last two columns have been removed


# Normalize the data
scaler = MinMaxScaler()
# Convert column names to strings
df.columns = df.columns.astype(str)

# Now apply the scaler
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)



# Step 2: Preprocess the Data
# Assuming the last column is the target variable
X = df_normalized.iloc[:, :-2]
y = labels
from sklearn.preprocessing import StandardScaler

# Fit on training data and transform it
X_scaled = scaler.fit_transform(X)
# print(X_scaled.shape)
test_data = pd.DataFrame(test_data)
# Identify columns with negative numbers
# negative_features = [col for col in test_data.columns if (test_data[col] < 0).any()]

# Drop these columns from test_data
test_data = test_data.drop(columns=negative_features)

# Transform the test data using the same scaler
test_data_scaled = pd.DataFrame(scaler.fit_transform(test_data), columns=test_data.columns)
# print(test_data_scaled.shape)
test_data_scaled = test_data_scaled.iloc[:, :-1]
# Step 2: Initialize PCA, choosing the number of components
pca = PCA(n_components=28)

# Step 3: Fit PCA on the standardized training data and transform it
X_pca = pca.fit_transform(X_scaled)

# Step 4: Transform the standardized test data using the same PCA transformation
X_test_data_pca = pca.transform(test_data_scaled)

# Step 5: Train RandomForestClassifier on the PCA-transformed training data
# rf_model = RandomForestClassifier(n_estimators=600, random_state=42)
# rf_model.fit(X_pca, y)
rf_model = MLPClassifier(hidden_layer_sizes=(800,400), solver='adam', random_state=42,max_iter=1200)
rf_model.fit(X_pca, y)

# Step 6: Make predictions on the PCA-transformed test data
test_predictions_pca = rf_model.predict(X_test_data_pca)

# Convert the NumPy array to a pandas DataFrame
predictions_df = pd.DataFrame(test_predictions_pca)
# print(predictions_df)
# Save the DataFrame to a CSV file
np.savetxt('predlabels.txt', test_predictions_pca, fmt='%d')
# print(f"Accuracy: {accuracy_score(test_labels, y_pred)}")

# Save the model to a file
joblib.dump(rf_model, 'model.pkl')