# Utilizing Semi Supervised Learning To Do Galaxy Morphological Classification
## Requirements
- Python 3.8.11
- torch == 1.9.0 + cu102
- scikit_learn == 0.24.2
- matplotlib == 3.4.2
- numpy == 1.19.5
- opencv-python == 4.5.3 <br>
You can utilize 'pip3 install -r requirements.txt' to 
## Abstract
In this study, in order to classify the pictures of galaxies automatically, we have
built a model of convolutional neural networks (CNN). With the optimized VGG-16
structure and semi-supervised machine learning strategy, the model has shown
surprising accuracy and efficiency in the differentiation between galaxies. The
datasets of galaxy pictures are from EFIGI and Galaxy Zoo 2 (GZ2). First, three
major galaxy types: Ellipticals (E), Spirals (Sc) and Irregular (I) have been well
classified. With 2,468 galaxy images as training data, the model can reach 94%
classification accuracy. Then, we expanded classification to 8 types: E, S0, Sa, Sb, Sc,
SBa, SBb, and SBc. With the help of semi-supervised strategy which using
autoencoder as pretrained model, data for model training was successfully enlarged
from 1,923 to 3,181 pictures, and the averaged classification accuracy finally
increased from 12.87% to 54.12%. Despite the similarity among spiral galaxies, the
model still can differentiate E (80%), Sb (80%), SBb (85%) and Sc (85%) patterns. In
conclusion, machine learning with semi-supervised strategy has been shown a better
solution when well-recognized galaxy patterns are not enough for model training, and
we hope the method could paved a new way for automatic astronomy patterns
recognition.
