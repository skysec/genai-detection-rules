"""
Negative test cases for PyTorch detection rule
These should NOT be detected by the detect-pytorch rule
"""

# Test 1: Comments mentioning torch
# This code uses torch but doesn't import it

# Test 2: Strings containing torch
framework = "torch"
description = "This uses PyTorch"

# Test 3: Dictionary with torch keys
config = {
    "framework": "pytorch",
    "device": "cuda"
}

# Test 4: Variable names
torch_enabled = True
pytorch_model = "model.pth"

# Test 5: URLs
docs_url = "https://pytorch.org/docs"

# Test 6: Environment variables
import os
TORCH_HOME = os.getenv("TORCH_HOME")

# Test 7: Mock tensor class
class Tensor:
    """Mock tensor - not PyTorch"""
    def __init__(self, data):
        self.data = data

    def cuda(self):
        return self

tensor = Tensor([1, 2, 3])
tensor.cuda()
