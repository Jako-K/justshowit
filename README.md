# Install

`pip install justshowit`

or 

`pip install git+https://github.com/Jako-K/justshowit`

# How to use
```python
from justshowit import show 
show(<your_image_source>)
```

# Demo


```python
from justshowit import show 
import cv2
import numpy as np
import torch
import random

# Example of different input images
url = "https://github.com/Jako-K/justshowit/blob/readme_stuff/test_image1.png"
numpy_image = np.random.rand(250, 400, 1)
torch_image = torch.ones((3, 300, 200)) * 255
image_bgr = cv2.imread("./test_images/test_image2.jpg")
path1 = "./readme_stuff/test_image2.jpg"
path2 = "./readme_stuff/test_image3.png"
path3 = "./readme_stuff/test_image4.png"
```


```python
# You can show a single image
image = show(path1)
```

![](https://github.com/Jako-K/justshowit/blob/main/readme_stuff/1.jpg?raw=True)


```python
# Do some basic, but very common stuff
show(image_bgr, resize_factor=0.50, BGR2RGB=True, save_to_path="./super_nice_image.png")
```

![](https://github.com/Jako-K/justshowit/blob/main/readme_stuff/2.png?raw=True)


```python
# Show a bunch of similiar sized images 
show([image_bgr, path1], resize_factor=0.5)
```

![](https://github.com/Jako-K/justshowit/blob/main/readme_stuff/3.png?raw=True)


```python
# An appropriate number of rows and columns will be chosen automatically
show([image_bgr for i in range(27)], BGR2RGB=True)
```

![](https://github.com/Jako-K/justshowit/blob/main/readme_stuff/4.png?raw=True)


```python
# You can also display a bunch of differently shaped images
show([url, torch_image, numpy_image, path1, path2, path3], resize_factor=0.5)
```

![](https://github.com/Jako-K/justshowit/blob/main/readme_stuff/5.png?raw=True)

```python
# A space effecient layout will be chosen automatically
random_images = [random.choice([url, torch_image, numpy_image, path1, path2, path3]) 
                 for _ in range(100)]
show(random_images)
```

![](https://github.com/Jako-K/justshowit/blob/main/readme_stuff/6.png?raw=True)
