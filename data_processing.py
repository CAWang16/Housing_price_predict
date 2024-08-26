
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import matplotlib.pyplot as plt

# Load the dataset
url = './Housing.csv'
data = pd.read_csv(url)

# Encode categorical features to numerical values
data['mainroad'] = data['mainroad'].map({'yes':1,'no':0})
data['guestroom'] = data['guestroom'].map({'yes':1,'no':0})
data['basement'] = data['basement'].map({'yes':1,'no':0})
data['hotwaterheating'] = data['hotwaterheating'].map({'yes':1,'no':0})
data['airconditioning'] = data['airconditioning'].map({'yes':1,'no':0})
data['prefarea'] = data['prefarea'].map({'yes':1,'no':0})
data['furnishingstatus'] = data['furnishingstatus'].map({'furnished':2,'semi-furnished':1,'unfurnished':0})

# Standardize the 'area' feature to be within the range [0, 1]
scaler = MinMaxScaler()
data['area'] = scaler.fit_transform(data[['area']])

# Split the data into training and testing sets
x = data.drop(columns=["price"])   # Features
y = data["price"]  # Target variable
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=87)

# Convert data to NumPy arrays
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Convert data to float64, because the model uses this precision
x_train = x_train.astype('float64')
x_test = x_test.astype('float64')
y_train = y_train.astype('float64')
y_test = y_test.astype('float64')

# Convert data to PyTorch tensors
x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

# Reshape target tensors to be compatible with the model's expected output
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# Define a linear regression model using PyTorch
class LinearRegressionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=x_train.shape[1], out_features=1, bias=True, dtype=torch.float64)
    def forward(self, x):
        return self.linear_layer(x)

# Set the seed for reproducibility
torch.manual_seed(87)
# Initialize the model
model = LinearRegressionModule()

# Define the loss function as Mean Squared Error (MSE)
cost_fn  = nn.MSELoss()

# Initial prediction and cost calculation
y_pred = model(x_train)
cost = cost_fn(y_pred, y_train)


# Define the optimizer using Stochastic Gradient Descent (SGD) with L2 regularization
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.002, weight_decay=1e-4)

print(model.state_dict()) # Print the current state of the model's parameters

# Define the number of epochs for training
Epochs = 50000

# Initialize lists to store the training and testing costs
train_cost_hist = []
test_cost_hist = []

# Initialize variables for early stopping
best_test_cost = float('inf') # Best observed test cost, initialized to infinity
patience, trials = 1000, 0  # Patience for early stopping and trial counter

for Epoch in range(Epochs):

    # Perform a forward pass and compute the training cost
    model.train()
    y_pred = model(x_train)
    train_cost = cost_fn(y_pred, y_train)
    train_cost_hist.append(train_cost.detach().numpy())
    optimizer.zero_grad() # Clear gradients from the previous step
    cost.backward()  # Compute the gradients
    optimizer.step() # Update the model parameters

    # Evaluate on the test set
    model.eval() # Set the model to evaluation mode
    with torch.inference_mode(): # Disable gradient computation
        test_pred = model(x_test) 
        test_cost = cost_fn(test_pred, y_test)
        test_cost_hist.append(test_cost.detach().numpy())  # Store the test cost
    
    # Early stopping condition
    if test_cost < best_test_cost:
        best_test_cost = test_cost
        trials = 0  # reset trials if we find a better model
    else:
        trials += 1
        if trials >= patience:
            print(f"Early stopping at epoch {Epoch}") # Stop training if no improvement for 'patience' epochs
            break

    # Print progress every 1000 epochs
    if Epoch % 1000 == 0:
        print(f'Epoch: {Epoch:5}   Train Cost: {train_cost:.4e}   Test Cost: {test_cost:.4e}')


# torch.save(obj=model.state_dict(), f='./pytorch_linear_regression2.pth')
# model1 = LinearRegressionModule()
# model1.state_dict()
# model1.load_state_dict(torch.load(f='./model/pytorch_linear_regression.pth'))

# Evaluate the model on the test set and compare predictions with actual values
model.eval()
with torch.inference_mode():
    y_pred = model(x_test)

y_pred = y_pred.squeeze()  # Remove extra dimensions if any
y_test = y_test.squeeze()  # Remove extra dimensions if any

# Calculate the absolute percentage error
percentage_errors = torch.abs((y_pred - y_test) / y_test) * 100

# Calculate the mean percentage error
mean_percentage_error = torch.mean(percentage_errors)

# Print the percentage errors and the mean percentage error
print(f'Mean percentage error: {mean_percentage_error:.2f}%')

# Plot the training and testing costs over epochs
plt.plot(range(len(train_cost_hist)), train_cost_hist, label='train cost')
plt.plot(range(len(test_cost_hist)), test_cost_hist, label='test cost')
plt.title('Train & Test Cost')
plt.xlabel("epochs")
plt.ylabel("cost")
plt.legend()
plt.show()