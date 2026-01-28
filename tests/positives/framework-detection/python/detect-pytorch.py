"""
Positive test cases for PyTorch detection rule
These should all be detected by the detect-pytorch rule
"""

# Test 1: Import torch
# ruleid: detect-pytorch
import torch

# Test 2: From torch import
# ruleid: detect-pytorch
from torch import nn, optim

# Test 3: Import torch.nn
# ruleid: detect-pytorch
import torch.nn

# Test 4: Import torch.optim
# ruleid: detect-pytorch
import torch.optim

# Test 5: Create tensor
# ruleid: detect-pytorch
tensor = torch.tensor([1, 2, 3])

# Test 6: Create Tensor
# ruleid: detect-pytorch
tensor = torch.Tensor([[1, 2], [3, 4]])

# Test 7: Zeros tensor
# ruleid: detect-pytorch
zeros = torch.zeros(3, 3)

# Test 8: Ones tensor
# ruleid: detect-pytorch
ones = torch.ones(2, 2)

# Test 9: Random tensor
# ruleid: detect-pytorch
rand_tensor = torch.rand(3, 3)

# Test 10: Random normal tensor
# ruleid: detect-pytorch
randn_tensor = torch.randn(2, 2)

# Test 11: Linear layer
# ruleid: detect-pytorch
linear = torch.nn.Linear(10, 5)

# Test 12: Conv2d layer
# ruleid: detect-pytorch
conv = torch.nn.Conv2d(3, 64, kernel_size=3)

# Test 13: LSTM layer
# ruleid: detect-pytorch
lstm = torch.nn.LSTM(10, 20, 2)

# Test 14: Transformer
# ruleid: detect-pytorch
transformer = torch.nn.Transformer()

# Test 15: ReLU activation
# ruleid: detect-pytorch
relu = torch.nn.ReLU()

# Test 16: CrossEntropyLoss
# ruleid: detect-pytorch
loss_fn = torch.nn.CrossEntropyLoss()

# Test 17: Adam optimizer
# ruleid: detect-pytorch
optimizer = torch.optim.Adam(model.parameters())

# Test 18: SGD optimizer
# ruleid: detect-pytorch
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Test 19: Save model
# ruleid: detect-pytorch
torch.save(model.state_dict(), 'model.pth')

# Test 20: Load model
# ruleid: detect-pytorch
state_dict = torch.load('model.pth')

# Test 21: Model train mode
# ruleid: detect-pytorch
model.train()

# Test 22: Model eval mode
# ruleid: detect-pytorch
model.eval()

# Test 23: No grad context
# ruleid: detect-pytorch
with torch.no_grad():
    output = model(input)

# Test 24: CUDA check
# ruleid: detect-pytorch
if torch.cuda.is_available():
    device = torch.device('cuda')

# Test 25: Move to CUDA
# ruleid: detect-pytorch
tensor.cuda()

# Test 26: DataLoader
# ruleid: detect-pytorch
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

# Test 27: Real-world model definition
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # ruleid: detect-pytorch
        self.fc1 = torch.nn.Linear(784, 128)
        # ruleid: detect-pytorch
        self.fc2 = torch.nn.Linear(128, 10)

    # ruleid: detect-pytorch
    def forward(self, x):
        # ruleid: detect-pytorch
        x = torch.nn.functional.relu(self.fc1(x))
        return self.fc2(x)

# Test 28: Training loop example
def train_model():
    # ruleid: detect-pytorch
    model = SimpleNet()
    # ruleid: detect-pytorch
    optimizer = torch.optim.Adam(model.parameters())
    # ruleid: detect-pytorch
    criterion = torch.nn.CrossEntropyLoss()

    # ruleid: detect-pytorch
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
