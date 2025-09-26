Used Car Price Prediction Model and Deployment FastAPIs.

This deep learning project follows a comprehensive Machine Learning pipeline to predict used car prices, emphasizing robust data preparation and a specialized Neural Network architecture using Keras Embeddings for enhanced performance on categorical features.

1. Data Cleaning & Preparation
Load and Initial Cleanup: Imported data and immediately dropped irrelevant columns (model, clean_title).

Missing Data Imputation: Handled missing values in fuel_type and accident by filling with the mode (most frequent value).

Type Conversion: Converted price and milage strings (containing symbols/units) into float numerical types.

2. Feature Engineering
Engine Data Extraction: Used Regular Expressions (Regex) on the raw engine column to extract three crucial numerical features: Horse Power, Engine Displacement, and Cylinder Count.

Engine Data Imputation: Filled missing horse_power and engine_displacement with the mean, and filled missing engine_cylinder with 0.

Categorical Normalization: Standardized the accident column to simple "Yes" or "No" values.

3. Data Transformation & Outlier Handling
Outlier Removal: Applied the Z-score method across all numerical columns, dropping any data points where the Z-score was greater than 3 (a standard deviation threshold).

Numerical Scaling: Used StandardScaler to normalize all numerical features (e.g., milage, model_year) to ensure equal contribution during training.

Categorical Encoding:

Label Encoding: Applied to low-cardinality features (fuel_type, accident).

Embedding Encoding: Applied LabelEncoder to high-cardinality features (brand, transmission, etc.) specifically to prepare them for the Embedding Layers in the neural network.

4. Deep Learning Model Architecture (Keras)
Functional API Structure: Utilized the Keras Functional API to combine two distinct input streams.

Specialized Inputs:

Embedding Inputs: Separate input and Embedding layer for each high-cardinality feature, followed by a Flatten layer to convert the embeddings into a single vector.

Numeric Input: A single input stream for all scaled numerical features.

Core Network: The concatenated features passed through two Dense layers (128 → 64) with 'relu' activation.

Regularization: Used L2 regularization on Dense layers and Dropout (0.3 and 0.2) after hidden layers to mitigate overfitting.

Output: A single neuron with 'linear' activation for the regression task (predicting a continuous price value).

Compilation: Optimized with the Adam optimizer and used Mean Absolute Error (MAE) as the loss function and primary metric.

Training Control: Implemented EarlyStopping (patience=5) to stop training once the validation loss ceased improving.

5. Training, Evaluation, and Deployment
Train-Test Split: Data was divided into 80% training and 20% testing sets.

Evaluation: Model performance was assessed using the MAE on both the training and testing sets to check for generalization.

Saving Assets: The trained Keras model (cars.keras), the StandardScaler, and all LabelEncoder objects were saved using pickle for persistence and deployment.

Deployment: The model and preprocessors were integrated into a FastAPI service for real-time price prediction.
