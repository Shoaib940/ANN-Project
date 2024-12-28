import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Data Transformation and Loading
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = torchvision.datasets.MNIST(
    root='./data', train=True, transform=data_transform, download=True)
test_data = torchvision.datasets.MNIST(
    root='./data', train=False, transform=data_transform, download=True)

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define the model using PyTorch's nn.Module
class SimpleNN(nn.Module):
    def _init_(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self)._init_()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Second fully connected layer

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten the input
        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = self.fc2(x)  # Output layer (logits)
        return x

# Model Initialization
input_dim = 28 * 28  # Flattened image size (28x28 pixels)
hidden_dim = 128
output_dim = 10  # 10 classes for MNIST digits
model = SimpleNN(input_dim, hidden_dim, output_dim)

# Optimizer and Loss Function
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Variables to store metrics for graphing
train_losses = []
train_accuracies = []
test_accuracies = []

# Training Loop
epochs = 10
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    cumulative_loss = 0
    correct_preds = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # Zero gradients before each backward pass
        
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backpropagate the gradients
        optimizer.step()  # Update weights and biases
        
        predictions = torch.argmax(outputs, dim=1)  # Get predicted labels
        correct_preds += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        cumulative_loss += loss.item()

    # Calculate epoch metrics
    avg_loss = cumulative_loss / len(train_loader)
    train_accuracy = 100 * correct_preds / total_samples
    train_losses.append(avg_loss)
    train_accuracies.append(train_accuracy)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

    # Evaluate on test data
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    test_accuracy = 100 * correct / total
    test_accuracies.append(test_accuracy)

    print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {test_accuracy:.2f}%")

# Plot Training and Test Metrics
plt.figure(figsize=(12, 6))

# Plot Training Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label="Training Loss", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid()

# Plot Training and Test Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, label="Training Accuracy", marker="o")
plt.plot(range(1, epochs + 1), test_accuracies, label="Test Accuracy", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training and Test Accuracy Over Epochs")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()










import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim

# Load and preprocess the California Housing dataset
data = fetch_california_housing()
inputs, targets = data.data, data.target

# Standardize inputs
scaler = StandardScaler()
inputs = scaler.fit_transform(inputs)

# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
train_features = torch.tensor(X_train, dtype=torch.float32)
train_targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
test_features = torch.tensor(X_test, dtype=torch.float32)
test_targets = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Prepare DataLoaders
train_data = DataLoader(TensorDataset(train_features, train_targets), batch_size=64, shuffle=True)
test_data = DataLoader(TensorDataset(test_features, test_targets), batch_size=64, shuffle=False)

# Define the Neural Network model
class SimpleNN(nn.Module):
    def _init_(self, input_size, hidden_size, output_size):
        super(SimpleNN, self)._init_()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Model parameters
input_size = train_features.shape[1]
hidden_size = 128
output_size = 1

# Instantiate model, loss function, and optimizer
model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Variables to store loss for plotting
train_losses = []
test_losses = []

# Training Loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for features, targets in train_data:
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Record average training loss for this epoch
    avg_train_loss = total_loss / len(train_data)
    train_losses.append(avg_train_loss)

    # Evaluate on the test dataset
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for features, targets in test_data:
            outputs = model(features)
            loss = criterion(outputs, targets)
            total_test_loss += loss.item() * targets.size(0)

    avg_test_loss = total_test_loss / len(test_data.dataset)
    test_losses.append(avg_test_loss)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

# Plot training and test losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label="Training Loss", marker="o")
plt.plot(range(1, epochs + 1), test_losses, label="Test Loss", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Test Loss Over Epochs")
plt.legend()
plt.grid()
plt.show()









import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc:.4f}')

# Plot training and validation accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='s')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()