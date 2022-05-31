# Utilizing Semi Supervised Learning To Do Galaxy Morphological Classification
## Requirements
- Python 3.8.11
- torch == 1.9.0 + cu102
- torchvision == 0.10.0+cu102
- scikit_learn == 0.24.2
- matplotlib == 3.4.2
- numpy == 1.19.5
- opencv-python == 4.5.3 <br/>

Dependencies can be installed using the following command: <br/>
```
pip3 install -r requirements.txt
```
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
## Data

## Usage
In our study, there are two tasks, the first task is about three type galaxy classification, the other is about eight type galaxy classification.<br/>
If you want to train a three type galaxy classifier, you can find out the source code in three_type folder.
```
python3 train_our_CNN.py // In this code, we construct the CNN on ourself.
python3 train_VGG16.py // In this code, we utilize VGG16 to be our CNN model.
```

## Reference
- Baillard, A., Bertin, E., De Lapparent, V., Fouqué, P., Arnouts, S., Mellier, Y., ... & Tasca, L. (2011). The EFIGI catalogue of 4458 nearby galaxies with detailed morphology. Astronomy & Astrophysics, 532, A74.
- Willett, K. W., Lintott, C. J., Bamford, S. P., Masters, K. L., Simmons, B. D., Casteels, K. R., ... & Thomas, D. (2013). Galaxy Zoo 2: detailed morphological classifications for 304 122 galaxies from the Sloan Digital Sky Survey. Monthly Notices of the Royal Astronomical Society, 435(4), 2835-2860.
- Vashishth, S., Yadav, P., Bhandari, M., & Talukdar, P. (2019, April). Confidence-based graph convolutional networks for semi-supervised learning. In The 22nd International Conference on Artificial Intelligence and Statistics (pp. 1792-1801). PMLR.
- LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., & Jackel, L. D. (1989). Backpropagation applied to handwritten zip code recognition. Neural computation, 1(4), 541-551.
