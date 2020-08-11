# Basic Operations on Images

This week you'll learn to:<br>

• Access pixel values and modify them<br>

• Access image properties<br>

• Setting Region of Image (ROI)<br>

• Splitting and Merging images<br>

Almost all the operations in this section is mainly related to Numpy rather than OpenCV.<br>

A good knowledge of Numpy is required to write better optimized code with OpenCV.<br>

# Accessing and Modifying pixel values

Let’s load a color image first:




```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
```


```python
img = cv2.imread('monalisa.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize = (75,7.5))
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()
```


![png](Week3_files/Week3_3_0.png)


You can access a pixel value by its row and column coordinates. <br>
For BGR image, it returns an array of Blue, Green,<br>
Red values. For grayscale image, just corresponding intensity is returned.<br>


```python
px = img[100,100]
print(px) 
# accessing only blue pixel
blue = img[100,100,0]
print(blue)

```

    [130 152 114]
    130


You can modify the pixel values the same way.



```python
img[100,100] = [255,255,255]
print( img[100,100])
```

    [255 255 255]


**Warning**: Numpy is a optimized library for fast array calculations. So simply accessing each and every pixel<br>
values and modifying it will be very slow and it is discouraged.<br>

**Note**: Above mentioned method is normally used for selecting a region of array, say first 5 rows and last 3 columns like<br>
that. For individual pixel access, Numpy array methods, **array.item()** and **array.itemset()** is considered to<br>
be better. But it always returns a scalar. So if you want to access all B,G,R values, you need to call array.item()<br>
separately for all<br>

Better pixel accessing and editing method :<br>



```python
# accessing RED value
print(img.item(10,10,2))

# modifying RED value
img.itemset((10,10,2),100)
print(img.item(10,10,2))

```

    54
    100


# Accessing Image Properties
Image properties include number of rows, columns and channels, type of image data, number of pixels etc.<br>
Shape of image is accessed by img.shape. It returns a tuple of number of rows, columns and channels (if image is<br>
color):<br>



```python
print(img.shape)
```

    (1788, 1200, 3)


**Note:** If image is grayscale, tuple returned contains only number of rows and columns.<br> 
So it is a good method tocheck if loaded image is grayscale or color image.

Total number of pixels is accessed by img.size:



```python
 print(img.size)
```

    6436800


Image datatype is obtained by img.dtype:


```python
print(img.dtype)
```

    uint8


**Note:** img.dtype is very important while debugging because a large number of errors in OpenCV-Python code is<br>
caused by invalid datatype.

# Image ROI
Sometimes, you will have to play with certain region of images. 

For eye detection in images, first perform face<br>
detection over the image until the face is found, then search within the face region for eyes.<br>
This approach improves accuracy (because eyes are always on faces :D ) and <br>
performance (because we search for a small area).<br>
ROI is again obtained using Numpy indexing. Here I am selecting Mona Lisa's face and copying it to another region in the<br>
image:<br>


```python
face = img[200:680, 400:780]
plt.figure(figsize = (75,7.5))
plt.imshow(face, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()
```


![png](Week3_files/Week3_19_0.png)


# Splitting and Merging Image Channels
The B,G,R channels of an image can be split into their individual planes when needed. Then, the individual channels<br>
can be merged back together to form a BGR image again. This can be performed by:<br>



```python
b,g,r = cv2.split(img)
img = cv2.merge((b,g,r))
#or
b = img[:,:,0]
```

Suppose, you want to make all the red pixels to zero, you need not split like this and put it equal to zero. You can<br>
simply use Numpy indexing which is faster.


```python
img[:,:,2] = 0
plt.figure(figsize = (75,7.5))
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()
```


![png](Week3_files/Week3_23_0.png)


**Warning:** cv2.split() is a costly operation (in terms of time), so only use it if necessary. Numpy indexing<br>
is much more efficient and should be used if possible.<br>


# Making Borders for Images (Padding)
If you want to create a border around the image, something like a photo frame, you can use **cv2.copyMakeBorder()**<br>
function. But it has more applications for convolution operation, zero padding etc. This function takes following<br>
arguments:<br>
•**src** - input image<br>

• **top, bottom, left, right** - border width in number of pixels in corresponding directions<br>

• **borderType** - Flag defining what kind of border to be added. It can be following types:<br>

– **cv2.BORDER_CONSTANT** - Adds a constant colored border. The value should be given as next<br>
       argument.<br>

– **cv2.BORDER_REFLECT** - Border will be mirror reflection of the border elements, like this :          fedcba|abcdefgh|hgfedcb<br>

– **cv2.BORDER_REFLECT_101 or cv2.BORDER_DEFAULT** - Same as above, but with a slight<br>
change, like this : gfedcb|abcdefgh|gfedcba<br>

– **cv2.BORDER_REPLICATE** - Last element is replicated throughout, like this:
aaaaaa|abcdefgh|hhhhhhh<br>

– **cv2.BORDER_WRAP** - Can’t explain, it will look like this : cdefgh|abcdefgh|abcdefg

• **value** - Color of border if border type is cv2.BORDER_CONSTANT<br>


Below is a sample code demonstrating all these border types for better understanding:<br>



```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
BLUE = [255,0,0]
img1 = cv2.imread('opencv-logo.PNG')
img1=cv2.resize(img1,(300,447))

replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)

reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)

reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)

wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)

constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

#Displaying all the images
plt.figure(figsize = (75,7.5))
plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.xticks([]), plt.yticks([])
plt.figure(figsize = (75,7.5))
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.xticks([]), plt.yticks([])
plt.figure(figsize = (75,7.5))
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.xticks([]), plt.yticks([])
plt.figure(figsize = (75,7.5))
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.xticks([]), plt.yticks([])
plt.figure(figsize = (75,7.5))
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.xticks([]), plt.yticks([])
plt.figure(figsize = (75,7.5))
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
plt.xticks([]), plt.yticks([]) 
plt.show()
```


![png](Week3_files/Week3_26_0.png)



![png](Week3_files/Week3_26_1.png)



![png](Week3_files/Week3_26_2.png)



![png](Week3_files/Week3_26_3.png)



![png](Week3_files/Week3_26_4.png)



![png](Week3_files/Week3_26_5.png)


# Arithmetic Operations on Images
Goal<br>
• Learn several arithmetic operations on images like addition, subtraction, bitwise operations etc.<br>
• You will learn these functions : **cv2.add()**, **cv2.addWeighted()** etc.<br>
Image Addition<br>
You can add two images by OpenCV function, **cv2.add()** or simply by numpy operation, res = img1 + img2.<br>
Both images should be of same depth and type, or second image can just be a scalar value.<br>

**Note:** There is a difference between OpenCV addition and Numpy addition. OpenCV addition is a saturated operation<br>
while Numpy addition is a modulo operation.<br>

For example, consider below sample:



```python
x = np.uint8([250])
y = np.uint8([10])
print(cv2.add(x,y))
print(x+y)
# 250+10 = 260 % 256 = 4
```

    [[255]]
    [4]


It will be more visible when you add two images. OpenCV function will provide a better result. 
<br>So always better stick
to OpenCV functions.

# Image Blending
This is also image addition, but different weights are given to images so that it gives a feeling of blending or transparency<br>
Images are added as per the equation below:<br>

𝑔(𝑥) = (1 − 𝛼)𝑓0(𝑥) + 𝛼𝑓1(𝑥)<br>

By varying 𝛼 from 0 → 1, you can perform a cool transition between one image to another.<br>

Here I took two images to blend them together. First image is given a weight of 0.7 and second image is given 0.3.<br>
cv2.addWeighted() applies following equation on the image.<br>

𝑑𝑠𝑡 = 𝛼 · 𝑖𝑚𝑔1 + 𝛽 · 𝑖𝑚𝑔2 + 𝛾<br>

Here 𝛾 is taken as zero.<br>


```python
img1 = cv2.imread('Messi.jpg')
img2 = cv2.imread('Ronaldo.jpg')

#Both images should be of same dimension. Thus resizing is necessary
img1= cv2.resize(img1,(900,650))
img2= cv2.resize(img2,(900,650))

dst = cv2.addWeighted(img1,0.5,img2,0.5,0)
cv2.imshow('Blended',dst)
cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

# Bitwise Operations
This includes bitwise AND, OR, NOT and XOR operations. They will be highly useful while extracting any part of<br>
the image (as we will see in coming chapters), defining and working with non-rectangular ROI etc. Below we will see<br>
an example on how to change a particular region of an image.<br>

I want to put OpenCV logo above an image. If I add two images, it will change color. If I blend it, I get an transparent<br>
effect. But I want it to be opaque. If it was a rectangular region, I could use ROI as we did in last chapter. But OpenCV<br>
logo is a not a rectangular shape. So you can do it with bitwise operations as below:<br>


```python
# Load two images
img1 = cv2.imread('Federer.jpg')
img1 = cv2.resize(img1,(500,400)) #Resizing 
cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

img2 = cv2.imread('Basketball.jpg')
img2 = cv2.resize(img2,(165,130))  #Resizing
img2 = img2[20:115,:]              #Cropping image to get only the basketball
cv2.imshow('res',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#I create a ROI on the federer image, i.e, the tennis ball
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]
# Now create a mask of basketball and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
# Now black-out the area of basketball in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
cv2.imshow('res1',img1_bg)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Take only region of basketball from basketball image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
cv2.imshow('res2',img2_fg)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Put basketball in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst
cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### The following code shows how the region surrounding the basketball in the final image is unaffected because of the bitwise operation


```python
# Load two images
img1 = cv2.imread('Federer1.jpeg')
img1 = cv2.resize(img1,(500,400))
cv2.imshow('Federer',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

img2 = cv2.imread('Basketball.jpg')
img2 = cv2.resize(img2,(160,125))
img2 = img2[20:110,:]
cv2.imshow('Basketball',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#I create a ROI on the federer image, i.e, the tennis ball
rows,cols,channels = img2.shape
roi = img1[0:rows, 100:100+cols ]
# Now create a mask of basketball and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
# Now black-out the area of basketball in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
cv2.imshow('res1',img1_bg)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Take only region of basketball from basketball image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
cv2.imshow('res2',img2_fg)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Put basketball in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 100:100+cols ] = dst
cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
```


```python

```
