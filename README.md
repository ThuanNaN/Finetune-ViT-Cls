# Vision Transformer for Image Classification from pre-trained model.

## 1. Data structure
```
data_dir/
    train/
        class_1/
        class_2/
        ...
    val/
        class_1/
        class_2/
        ...
    test/
        class_1/
        class_2/
        ...
```

## 2. Install packages
```
pip install -r requirements.txt
```

## 3. Config
Look example of data config and training config in config folder

## 4. Run
Just run this command:
```
python train.py
```

# NOTE
To faster training on Apple M1 Silicon using GPU with Pytorch:
```
opt.device = "cuda" if torch.cuda.is_available() else "cpu"
# replace to
opt.device = torch.device("mps")
```
