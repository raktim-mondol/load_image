# How to load images for Segmentation and Classification?
<span style="color:red">some *red* text</span>

#### Install Required Libraries 
pip install --upgrade pip\
pip install os\
pip install cv2==4.5.1\
pip install numpy\
pip install tqdm\
pip install scikit-image\
pip install keras\
pip install matplotlib

## Directory Layout
### Segmentation Folder Structure

```bash
├── train_folder
│   ├── data_folder
│   │   ├── image
│   │    ├── 1.tif
│   │    ├── 2.tif
│   │    ├── 3.tif
│   │   ├── mask
│   │    ├── 1.tif
│   │    ├── 2.tif
│   │    ├── 3.tif

OR

├── train_folder
│   ├──sample1
│   │   ├── image
│   │    ├── 1.tif
│   │    ├── 2.tif
│   │    ├── 3.tif
│   │   ├── mask
│   │    ├── 1.tif
│   │    ├── 2.tif
│   │    ├── 3.tif
│   ├── sample2
│   │   ├── image
│   │    ├── 1.tif
│   │    ├── 2.tif
│   │    ├── 3.tif
│   │   ├── mask
│   │    ├── 1.tif
│   │    ├── 2.tif
│   │    ├── 3.tif

```
### Classifcation Folder Structure 
```
├── train_folder
│   ├── monkey
│   │    ├── 1.tif
│   │    ├── 2.tif
│   │    ├── 3.tif
│   ├── tiger
│   │    ├── 1.tif
│   │    ├── 2.tif
│   │    ├── 3.tif
│   ├── dog
│   │    ├── 1.tif
│   │    ├── 2.tif
│   │    ├── 3.tif
```

### Data Load for Segmentation
```python

from data_class import Data
IMAGE_PATH = 'C:/Users/example/train_folder/'

data_obj= Data()
# You can provide your desired image size
# You can upscale or downscale

X_train, Y_train = data_obj.load_segmentation_data(IMAGE_PATH, 'tif', 256, 256)

# Visualize image with corresponding mask
data_obj.visualize(X_train,Y_train)

```
### Data Load for Classification 

```python

from data_class import Data
IMAGE_PATH = 'C:/Users/example/train_folder/'

data_obj= Data()
# You can provide your desired image size
# You can upscale or downscale
# to_catagorical=True will return one hot encoded value to Y_train
X_train, Y_train = data_obj.load_classification_data(IMAGE_PATH, 'tif', 256, 256, to_cat=True)


#Check Data Label
data_obj.label_check()

# Visualize image with corresponding mask
data_obj.visualize(X_train)

```



