# Puyo Color Classification Using a CNN

## PyTorch Model
```python
class PuyoClassifier(nn.Module):
    def __init__(self):
        super().__init__()        

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3), # -> [-1, 32, 28, 30]
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3), # -> [-1, 16, 26, 28]
            nn.MaxPool2d(2) # -> [-1, 16, 13, 14]
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 13 * 14, 7),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 16 * 13 * 14)
        x = self.fc(x)
        return x
```
You can load the saved model weights from `puyo-classifier.pt`.