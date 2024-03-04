import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

#finds the principle components and graphs the 2 most significant
pca_data = np.dot(mean_numbers, eig_vec)

# Feature Scaling
features = [col for col in data.columns if col != 'Target']
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# PCA for feature reduction
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_scaled)
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Prepare final dataset for training
X = principal_df
y = data['Target'].values  # Assuming 'Target' is your target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network Model with Regularization
def build_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),  # Dropout layer to prevent overfitting
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = build_model(X_train.shape[1])

# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Model Training
model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping])

# Evaluate Model
loss = model.evaluate(X_test, y_test)
print(f'Mean Squared Error on Test Set: {loss}')

# Predictions
# For new predictions, ensure you scale and PCA-transform the new data similarly
# new_data_scaled = scaler.transform(new_data)
# new_data_pca = pca.transform(new_data_scaled)
# predictions = model.predict(new_data_pca)

# Print or use the predictions as needed
