# Linear Regression Model using PyTorch

This project implements a linear regression model from scratch using **PyTorch**. The model is trained to predict housing prices based on various features such as road access, guest rooms, and more. It demonstrates the use of gradient descent optimization (SGD) and early stopping for improving model performance. The dataset used is assumed to be in a CSV format, titled `Housing.csv`.

## Features of the Project

- **Data Preprocessing:**
  - Categorical features are encoded as numerical values for use in the regression model.
  - The area feature is standardized using `MinMaxScaler` to ensure the values range between [0,1].
  
- **Model:**
  - The model is a simple linear regression model built using `torch.nn.Module`.
  - Loss function: Mean Squared Error (MSE).
  - Optimizer: Stochastic Gradient Descent (SGD) with L2 regularization.
  
- **Training Process:**
  - The model is trained over 50,000 epochs, with early stopping based on test set performance.
  - Both training and testing costs are tracked and plotted for visualization.

- **Performance Evaluation:**
  - The Mean Percentage Error is calculated to evaluate prediction accuracy.
  - Training and testing costs are visualized over the epochs.
