import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import os

# Define the neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Training function
def train(model, train_loader, optimizer, criterion, device, log_file):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Batch: {batch_idx}/{len(train_loader)} Loss: {loss.item():.6f}')
            with open(log_file, 'a') as f:
                f.write(f'Batch: {batch_idx}, Loss: {loss.item():.6f}\n')

def main(learning_rate):
    # Set up logging
    log_file = f'/n/holyscratch01/hankyang_lab/workshop/training_log_lr_{learning_rate}.txt'
    os.makedirs(os.path.dirname(log_file), exist_ok=True) 
    with open(log_file, 'w') as f:
        f.write(f'Training log for learning rate: {learning_rate}\n')

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train(model, train_loader, optimizer, criterion, device, log_file)

    print(f'Training complete. Log file: {log_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST Training Script')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()

    main(args.lr)