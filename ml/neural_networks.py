import pandas as pd
import numpy as np
import joblib

# Force TensorFlow to use only the CPU to avoid CUDA errors
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

path = '../datasets/data_4000v/env_vital_signals.txt'

try:
    df = pd.read_csv(path, header=None)
    df.columns = [
        'vic_id', 'sistolic_presion', 'diastolic_pression', 'qPA', 
        'pulse', 'resp_freq', 'gravity_value', 'gravity_class'
    ]
except FileNotFoundError:   
    print("Error: Dataset file not found. Path: " + path)
    exit()

X = df[['qPA', 'pulse', 'resp_freq']].values
y_reg = df['gravity_value'].values
y_class = (df['gravity_class'] - 1).values

# --- SCALING IS CRUCIAL FOR NEURAL NETWORKS ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'data_scaler.joblib') # Save scaler for use in the rescuer agent
print("Data loaded and scaled successfully.")
print("-" * 50)

def create_model(config, task='classifier', input_shape=None):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    
    # Add hidden layers based on the config
    for layer_config in config['layers']:
        model.add(Dense(layer_config['neurons'], activation=layer_config['activation']))
    
    # Add dropout if specified
    if config.get('dropout_rate'):
        model.add(Dropout(config['dropout_rate']))
        
    # Optimizer configuration
    optimizer_name = config.get('optimizer', 'adam')
    learning_rate = config.get('learning_rate', 0.001)
    
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else: # Default to Adam
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if task == 'classifier':
        model.add(Dense(4, activation='softmax'))
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else: # Regressor
        model.add(Dense(1))
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
    return model

# Define a list of configurations to test
nn_configs = [
    {
        "name": "Simple_ReLU",
        "layers": [{"neurons": 32, "activation": "relu"}],
        "optimizer": "adam",
        "learning_rate": 0.001
    },
    {
        "name": "Deeper_ReLU_HighLR",
        "layers": [
            {"neurons": 64, "activation": "relu"},
            {"neurons": 32, "activation": "relu"}
        ],
        "optimizer": "adam",
        "learning_rate": 0.01
    },
    {
        "name": "Wider_Tanh_With_Dropout",
        "layers": [{"neurons": 128, "activation": "tanh"}],
        "dropout_rate": 0.3,
        "optimizer": "rmsprop",
        "learning_rate": 0.001
    },
    {
        "name": "Deeper_HighDropout",
        "layers": [
            {"neurons": 64, "activation": "relu"},
            {"neurons": 32, "activation": "relu"}
        ],
        "dropout_rate": 0.5,
        "optimizer": "adam",
        "learning_rate": 0.001
    },
]


# --- 3. Train and Evaluate Regressors ---
print("--- Evaluating Neural Network REGRESSORS ---")
for config in nn_configs:
    print(f"\nTraining Regressor Configuration: {config['name']}")
    fold_mses = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, val_index in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_index], X_scaled[val_index]
        y_train, y_val = y_reg[train_index], y_reg[val_index]
        
        model = create_model(config, task='regressor', input_shape=X_scaled.shape[1])
        model.fit(X_train, y_train, epochs=50, verbose=0)
        mse = model.evaluate(X_val, y_val, verbose=0)
        fold_mses.append(mse)
    
    rmse_scores = np.sqrt(fold_mses)
    print(f"  RMSE (5-fold CV): {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")

print("\n" + "-" * 50)

# --- 4. Train and Evaluate Classifiers ---
print("--- Evaluating Neural Network CLASSIFIERS ---")
for config in nn_configs:
    print(f"\nTraining Classifier Configuration: {config['name']}")
    fold_accs, fold_precs, fold_recs = [], [], []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, val_index in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_index], X_scaled[val_index]
        y_train, y_val = y_class[train_index], y_class[val_index]
        
        model = create_model(config, task='classifier', input_shape=X_scaled.shape[1])
        model.fit(X_train, y_train, epochs=50, verbose=0)
        
        y_pred_probs = model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        fold_accs.append(accuracy_score(y_val, y_pred))
        fold_precs.append(precision_score(y_val, y_pred, average='macro', zero_division=0))
        fold_recs.append(recall_score(y_val, y_pred, average='macro', zero_division=0))
        
    print(f"  Accuracy:  {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")
    print(f"  Precision: {np.mean(fold_precs):.4f} (+/- {np.std(fold_precs):.4f})")
    print(f"  Recall:    {np.mean(fold_recs):.4f} (+/- {np.std(fold_recs):.4f})")

print("\n" + "=" * 50)
print("Experimentation complete.")
print("Choose the best configuration dictionary from the list above.")
print("Then, uncomment the lines below to train it on all data and save it.")
print("=" * 50)

# CHOOSE YOUR BEST CONFIGURATION DICTIONARY HERE

best_regressor_config = nn_configs[1]
best_classifier_config = nn_configs[1]

# Train final regressor on all data
final_regressor = create_model(best_regressor_config, task='regressor', input_shape=X_scaled.shape[1])
final_regressor.fit(X_scaled, y_reg, epochs=50, verbose=0) 
final_regressor.save('best_regressor_nn.h5')
print("\nSaved best NN regressor to 'best_regressor_nn.h5'")

# Train final classifier on all data
final_classifier = create_model(best_classifier_config, task='classifier', input_shape=X_scaled.shape[1])
final_classifier.fit(X_scaled, y_class, epochs=50, verbose=0)
final_classifier.save('best_classifier_nn.h5')
print("Saved best NN classifier to 'best_classifier_nn.h5'")
