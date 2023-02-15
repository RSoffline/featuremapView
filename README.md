# featuremapView
CNNモデルの中間特徴量を表示する

# 使用例
```python
from controller import Controller
from torchvision import transforms

model = your_model()
preprocess = transforms.Compose([
    lambda img:img.convert("RGB"),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
Controller(model, "model", device, preprocess).run()
```