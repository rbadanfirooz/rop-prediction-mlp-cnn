# ROP Prediction Using Deep Learning

Predicting the Rate of Penetration (ROP) in drilling operations using deep learning models, including a Multilayer Perceptron (MLP) and a One-Dimensional Convolutional Neural Network (1D-CNN).

## Overview

The Rate of Penetration (ROP) is an important drilling performance indicator that measures the speed at which a drill bit penetrates a formation. Accurate ROP prediction can support drilling performance analysis, operational planning, and data-driven decision-making.

This project implements and compares two neural network architectures to predict ROP from drilling operational parameters and mud logging data.

## Models

### 1. Multilayer Perceptron (MLP)

A feedforward neural network designed to learn nonlinear relationships between drilling parameters and ROP.

**Architecture:**
- Input layer
- Dense layer with 128 neurons and ReLU activation
- Dropout layer (0.2)
- Dense layer with 64 neurons and ReLU activation
- Dropout layer (0.15)
- Dense layer with 32 neurons and ReLU activation
- Linear output layer for regression

### 2. One-Dimensional Convolutional Neural Network (1D-CNN)

A convolutional neural network that applies one-dimensional convolution operations to the input feature vector to learn patterns among the input features.

**Architecture:**
- 1D convolutional layer with 64 filters
- Max pooling layer
- 1D convolutional layer with 32 filters
- Global average pooling
- Dense layer with 32 neurons
- Linear output layer for regression

The CNN treats the input features as a one-dimensional sequence. This design does not, by itself, imply that the dataset contains a temporal sequence.

## Dataset

The project uses an Excel dataset named `MudLogging_Data.xlsx`.

The dataset contains drilling-related parameters, including:

| Feature | Description | Unit |
|---|---|---|
| SPP | Standpipe Pressure | psi |
| ROP | Rate of Penetration (target variable) | m/h |
| PUMP 1 | Pump 1 operating parameter | spm |
| PUMP 2 | Pump 2 operating parameter | spm |
| PUMP 3 | Pump 3 operating parameter | spm |
| FLWpmps | Flow rate | gpm |
| DESANDEER | Desander-related volume measurement | bbl |
| RESERVE 1 | Reserve tank volume | bbl |
| SUCTION 1 | Suction tank volume | bbl |
| SUCTION2 | Suction tank volume | bbl |
| MIDDLE | Middle tank volume | bbl |
| MW IN | Inlet mud weight | kg/l |
| Overpull | Overpull measurement | klbs |
| Vol+/- | Volume change measurement | bbl |

*Note: The descriptions and units above follow the provided sample. The exact operational meaning of individual measurements should be verified against the original data documentation.*

## Features

- Excel data loading and preprocessing using Pandas
- Handling of missing and non-numeric values
- Feature scaling using `StandardScaler`
- Training, validation, and test dataset splitting
- MLP and 1D-CNN model implementations using TensorFlow/Keras
- Early stopping to help reduce overfitting
- Model checkpointing to retain the best validation weights
- Model evaluation using regression metrics
- Visualization of training history and prediction performance
- Saving trained models and the fitted scaler

## Data Preprocessing

The preprocessing pipeline includes:

1. Loading the Excel dataset.
2. Removing rows containing bracketed text.
3. Converting feature values to numeric representations.
4. Removing configured columns when present.
5. Identifying the target column, `ROP`.
6. Handling missing target values and missing feature values.
7. Splitting the data into training, validation, and test sets.
8. Standardizing features using a scaler fitted on the training set.

The configured split is approximately:
- Training: 70%
- Validation: 15%
- Testing: 15%

For rigorous evaluation, imputation statistics should be learned from the training set only, and the dataset splitting strategy should account for temporal or well-level dependencies when applicable.

## Evaluation Metrics

The models are evaluated using the following metrics:

- **RMSE (Root Mean Squared Error):** Measures prediction error while penalizing larger errors more strongly.
- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted ROP values.
- **R² (Coefficient of Determination):** Measures how well the model explains the variance in the target values.

Lower RMSE and MAE values generally indicate better predictive performance. An R² value closer to 1 indicates a better fit, although results depend on the dataset and evaluation methodology.

## Requirements

- Python 3.10 or later (use a version compatible with the installed TensorFlow release)
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow
- OpenPyXL
- Joblib

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/rop-prediction-mlp-cnn.git
cd rop-prediction-mlp-cnn
```

Create and activate a virtual environment:

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Place `MudLogging_Data.xlsx` in the project root directory, alongside `main.py`.

The script expects the target column to be named:

```text
ROP
```

Ensure that the dataset is available and that its column names match the expected configuration.

## Usage

Run the main script:

```bash
python main.py
```

The script will:

1. Load and preprocess the dataset.
2. Train the MLP model.
3. Evaluate the MLP model on the test set.
4. Display training and prediction plots.
5. Prepare the data for the 1D-CNN model.
6. Train and evaluate the 1D-CNN model.
7. Print a comparison of both models.
8. Save the trained models and scaler when saving succeeds.

Training parameters such as the number of epochs, batch size, learning rate, and early stopping patience can be adjusted in `main.py`.

## Output Files

Depending on the training run, the script generates:

| File | Description |
|---|---|
| `mlp_best.h5` | Best MLP weights |
| `cnn1d_best.h5` | Best 1D-CNN weights |
| `mlp_model.h5` | Saved MLP model |
| `cnn1d_model.h5` | Saved 1D-CNN model |
| `scaler.save` | Fitted feature scaler |

Training plots are displayed during execution. They can be saved to disk by extending the plotting functions.

## Model Comparison

The project compares the MLP and 1D-CNN using the same train, validation, and test partitions.

The comparison table reports:

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| MLP | Generated during execution | Generated during execution | Generated during execution |
| 1D-CNN | Generated during execution | Generated during execution | Generated during execution |

Actual performance metrics should be recorded after running the models on the dataset. No performance advantage is assumed in advance.

## Limitations and Future Improvements

Potential improvements include:

- Using a scikit-learn preprocessing pipeline to prevent data leakage during imputation and scaling.
- Comparing random splitting with time-based or well-based validation.
- Performing hyperparameter optimization.
- Comparing neural networks with baseline models such as Linear Regression, Random Forest, and Gradient Boosting.
- Investigating feature importance and prediction interpretability.
- Saving evaluation metrics and plots automatically.
- Testing the trained models on independent drilling data.

## Reproducibility

The project sets random seeds for Python, NumPy, and TensorFlow. However, exact reproducibility may still depend on hardware, software versions, and nondeterministic operations.

## Disclaimer

This project is intended for educational, research, and experimental purposes. Predictions should not be used as the sole basis for operational drilling decisions without independent validation and appropriate engineering review.

## License

Choose and add a suitable open-source license before distributing the code for reuse. Confirm that the dataset's terms also permit redistribution.

## Author

**YOUR NAME**

GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)

