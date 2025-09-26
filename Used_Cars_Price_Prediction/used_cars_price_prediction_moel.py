import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy import stats
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2
import pickle

# ===========================
# 1. Importing and Inspecting Data
# ===========================
car = pd.read_csv("used_cars.csv")
print(car.head())
print(car.dtypes)
print(car.isnull().sum())

# ===========================
# 2. Data Cleaning
# ===========================
# Drop irrelevant columns
car.drop(columns=["model", "clean_title"], inplace=True)

# Handle missing values by filling with mode
car["fuel_type"] = car["fuel_type"].fillna(car["fuel_type"].mode()[0])
car["accident"] = car["accident"].fillna(car["accident"].mode()[0])
print(car.isnull().sum())

# Convert string-based numerical columns into floats
car["milage"] = car["milage"].str.replace("mi", "").str.replace(",", "").str.replace(" ", "").astype(float)
car["price"] = car["price"].str.replace("$", "").str.replace(",", "").str.replace(" ", "").astype(float)

# ===========================
# 3. Feature Engineering
# ===========================
# Extract horsepower, engine displacement, and cylinder count
car["horse_power"] = car["engine"].str.extract(r'(\d+\.?\d*)\s*HP').astype(float)
car["engine_displacement"] = car["engine"].str.extract(r'(\d+\.?\d*)\s*L').astype(float)
car["engine_cylinder"] = car["engine"].str.extract(r'(\d+)\s*Cylinder').astype(float)

# Fill missing values for engineered features
car["horse_power"] = car["horse_power"].fillna(car["horse_power"].mean())
car["engine_displacement"] = car["engine_displacement"].fillna(car["engine_displacement"].mean())
car["engine_cylinder"] = car["engine_cylinder"].fillna(0).astype(int)

# Drop the original 'engine' column after extraction
car.drop(columns="engine", inplace=True)

# Normalize accident column into Yes/No
car["accident"].replace("At least 1 accident or damage reported", "Yes", inplace=True)
car["accident"].replace("None reported", "No", inplace=True)

# ===========================
# 4. Outlier Removal
# ===========================
outliers_num_cols = car.select_dtypes(exclude="object").columns
z_score = np.abs(stats.zscore(car[outliers_num_cols]))
cars = car[(z_score < 3).all(axis=1)]
print(f"{car.shape}\n{cars.shape}")
print(cars.isnull().sum())
print(cars.dtypes)

# ===========================
# 5. Scaling Numerical Columns
# ===========================
num_columns = ["model_year", "milage", "horse_power", "engine_displacement", "engine_cylinder"]
scaling = StandardScaler()
cars[num_columns] = scaling.fit_transform(cars[num_columns])

target_column = "price"
target_scaling = StandardScaler()
cars[target_column] = target_scaling.fit_transform(cars[[target_column]])

# ===========================
# 6. Encoding Categorical Columns
# ===========================
cat_cols = cars.select_dtypes(include="object").columns
for col in cat_cols:
    print(f"{col} include {cars[col].nunique()}")

# Apply LabelEncoding for specific binary-like features
label_cols = ["fuel_type", "accident"]
encoders = {}
for col in label_cols:
    label = LabelEncoder()
    cars[col] = label.fit_transform(cars[col])
    encoders[col] = label

# Apply LabelEncoding for embedding-based categorical features
emb_cols = ["brand", "transmission", "ext_col", "int_col"]
emb_encoders = {}
for col in emb_cols:
    emb_label = LabelEncoder()
    cars[col] = emb_label.fit_transform(cars[col])
    emb_encoders[col] = emb_label

print(cars.dtypes)

# ===========================
# 7. Build Neural Network with Embeddings
# ===========================
emb_inputs, emb_layers = [], []
emb_cols = ["brand", "transmission", "ext_col", "int_col"]

# Create embedding layers for high-cardinality categorical features
for col in emb_cols:
    vocab_size = cars[col].nunique() + 1
    emb_dim = min(100, vocab_size // 2)

    inputs = Input(shape=(1,), name=f"{col}_input")
    emb = Embedding(input_dim=vocab_size, output_dim=emb_dim, name=f"{col}_emb")(inputs)
    emb = Flatten()(emb)
    emb_inputs.append(inputs)
    emb_layers.append(emb)

# Concatenate embeddings + numeric inputs
x = Concatenate()(emb_layers)
num_cols = ["model_year", "milage", "horse_power", "engine_displacement", "engine_cylinder"]
num_inputs = Input(shape=(len(num_cols),), name="num_inputs")
x = Concatenate()([x, num_inputs])

# Dense layers with regularization and dropout
x = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(x)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x)
x = Dropout(0.2)(x)
output = Dense(1, activation="linear", kernel_regularizer=l2(0.01))(x)

# Final model with all inputs
all_inputs = emb_inputs + [num_inputs]
model = Model(inputs=all_inputs, outputs=output)

# Compile model with early stopping
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model.compile(optimizer=Adam(learning_rate=0.0001), loss="mae", metrics=["mae"])

# ===========================
# 8. Train-Test Split
# ===========================

X = cars.drop(columns= target_column)
y = cars[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Prepare model inputs (embedding + numeric features)
X_train_inputs = [X_train[col].values for col in emb_cols] + [X_train[num_cols].values]
X_test_inputs  = [X_test[col].values for col in emb_cols] + [X_test[num_cols].values]

# ===========================
# 9. Model Training
# ===========================
history = model.fit(
    X_train_inputs, y_train,
    validation_data=(X_test_inputs, y_test),
    epochs=32,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# ===========================
# 10. Model Evaluation
# ===========================
train_mae = model.evaluate(X_train_inputs, y_train, verbose=0)[1]
test_mae  = model.evaluate(X_test_inputs, y_test, verbose=0)[1]

print(f"Train MAE: {train_mae:.2f}")
print(f"Test MAE: {test_mae:.2f}")

# ===========================
# 11. Save Model & Preprocessors
# ===========================
model.save("cars.keras")

with open("label.pkl", "wb") as f:
    pickle.dump(encoders, f)
with open("emb_label.pkl", "wb") as f:
    pickle.dump(emb_encoders, f)
with open("scale.pkl", "wb") as f:
    pickle.dump(scaling, f)
with open("target_scale", "wb") as f:
    pickle.dump(target_scaling, f)
print("✅ Model and preprocessors saved successfully!")
