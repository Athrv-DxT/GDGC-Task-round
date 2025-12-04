# MURA X-Ray Classification (Normal vs Abnormal)

## What this project basically does

- Unzips the MURA dataset from Google Drive in Colab  
- Loads the CSV files (`train_image_paths.csv` and `valid_image_paths.csv`)  
- Builds full image paths and extracts labels from the path names  
- Preprocesses the X-rays (grayscale → resize → normalize)  
- Trains a MobileNetV2 model to classify normal vs abnormal  
- Shows loss/accuracy curves and some sample predictions  

---

## ⚙️ Preprocessing

For each image, I did:

- Converted to **grayscale**  
- Resized to **224x224**  
- Added some basic augmentation on training images  
  - random rotation  
  - random horizontal flip  
- Normalized using ImageNet mean/std (MobileNet expects that)

---

## Model & Training (MobileNetV2 fine-tuning)

I used MobileNetV2 with pretrained ImageNet weights and swapped the final layer with a single output unit since this is binary classification.

Loss → `BCEWithLogitsLoss`  
Optimizer → Adam (lr = 1e-4)  
Batch size → 32  
Epochs → 3–5 depending on runtime  

**Link to google colab :** https://colab.research.google.com/drive/1JDKo0W3HBcFJ-fZwtHAFlJVhggMPLv26?usp=sharing

