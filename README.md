# CAPTCHA Recognizer

## Usage

Install requirements:
```bash
pip install timm safetensors pillow
```

Predict CAPTCHA:
```bash
python cap.py --model_path <path_to_ckpt> --image_path <path_to_image1> <path_to_image2>
```

## Train

1. Install RainbowNeko Engine:
```bash
pip install rainbowneko>=1.8
```

2. Setup `data_root` in `cfgs/py/train/captcha.py`

3. Train with RainbowNeko Engine:
```bash
neko_train_1gpu --cfg cfgs/py/train/captcha.py
```