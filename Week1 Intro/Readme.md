# Image Processing using OpenCV

## -By IEEE NITK

### Veiwing Images

In this snippet, we load a image and we then display the same image If this works without any issues you're good to go!


```python
import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('First.jpeg',1)
print(img)
cv2.imshow('image', img)
cv2.waitKey()
cv2.destroyAllWindows()
```

    [[[123  29  10]
      [149  55  36]
      [150  53  33]
      ...
      [171  67  98]
      [157  56  81]
      [153  54  74]]
    
     [[136  42  23]
      [151  58  37]
      [142  45  25]
      ...
      [167  63  94]
      [161  61  83]
      [159  60  78]]
    
     [[151  58  37]
      [156  63  42]
      [137  40  20]
      ...
      [158  56  84]
      [161  60  81]
      [160  61  77]]
    
     ...
    
     [[166  23  45]
      [164  23  44]
      [162  20  43]
      ...
      [134  27  23]
      [139  30  28]
      [141  30  28]]
    
     [[164  27  48]
      [161  24  45]
      [159  22  44]
      ...
      [139  32  28]
      [136  25  23]
      [138  25  23]]
    
     [[151  25  44]
      [151  25  44]
      [151  24  45]
      ...
      [140  33  26]
      [138  25  22]
      [142  27  24]]]


## Conversion From RGB to a Grayscale Image:¶

Here we use the formulae descirbed earlier to do the conversion Here's the formula again:

### Average
Gray = (R+G+B)/3

###  Luminosity
Gray = 0.21R + 0.72G + 0.07*B


```python
blue = img[:,:,0]
green = img[:,:,1]
red = img[:,:,2]

mean = (0.33*blue + 0.33*green + 0.33*red).astype('uint8')
lumin = (0.07*blue + 0.72*green + 0.21*red).astype('uint8')


cv2.imshow('mean',mean)
#cv2.waitKey()
#cv2.destroyAllWindows()

cv2.imshow('lumin',lumin)
cv2.waitKey()
cv2.destroyAllWindows()
```

### Conversion from Grayscale to Binary:

Here we will implement the concept of thresholding

Here, the matter is straight forward.

If pixel value is greater than a threshold value, it is assigned one value (may be white), else it is assigned another value (may be black).

The function used is cv2.threshold.

First argument is the source image, which should be a grayscale image.

Second argument is the threshold value which is used to classify the pixel values.

Third argument is the maxVal which represents the value to be given if pixel value is more than (sometimes less than) the threshold value.

OpenCV provides different styles of thresholding and it is decided by the fourth parameter of the function.

cv2.THRESH_BINARY

cv2.THRESH_BINARY_INV

cv2.THRESH_TRUNC

cv2.THRESH_TOZERO

cv2.THRESH_TOZERO_INV

All values above the threshold of 127 are set high (255) , and those below are set low (0)


```python
img = cv2.imread('threshold.jpg',0) #Threshold_adaptive
ret,th1 = cv2.threshold(img,140,255,cv2.THRESH_BINARY)
cv2.imshow('output',th1)
cv2.waitKey()
cv2.destroyAllWindows()
```

In the previous section, we used a global value as threshold value. But it may not be good in all the conditions where image has different lighting conditions in different areas.


```python
img = cv2.imread('Threshold_adaptive.jpeg',0)

th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,2)       #Mean
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)   #Gaussian
cv2.imshow('out',th2)
cv2.imshow('output',th3)
cv2.waitKey()
cv2.destroyAllWindows()
```


## Detection of a Specific Color in an Image:


```python
img = cv2.imread('gems.jpg',1)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower_color = np.array([120, 60,60 ]) 
upper_color = np.array([170, 220, 190])
mask = cv2.inRange(hsv, lower_color, upper_color)
out = cv2.bitwise_and(img,img,mask=mask)
cv2.imshow('original',img)
cv2.imshow('output',out)
cv2.waitKey()
cv2.destroyAllWindows()
```

### Read the image "nowYouDont.png" and see what is written on it...........


```python
import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('nowYouDont.png',1)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.imshow('original',hsv)
print(img)
cv2.waitKey()
cv2.destroyAllWindows()

# Write code here
```

    [[[ 32  32 145]
      [ 32  32 145]
      [ 32  32 145]
      ...
      [ 32  32 145]
      [ 32  32 145]
      [ 32  32 145]]
    
     [[ 32  32 145]
      [ 32  32 145]
      [ 32  32 145]
      ...
      [ 32  32 145]
      [ 32  32 145]
      [ 32  32 145]]
    
     [[ 32  32 145]
      [ 32  32 145]
      [ 32  32 145]
      ...
      [ 32  32 145]
      [ 32  32 145]
      [ 32  32 145]]
    
     ...
    
     [[ 32  32 145]
      [ 32  32 145]
      [ 32  32 145]
      ...
      [ 32  32 145]
      [ 32  32 145]
      [ 32  32 145]]
    
     [[ 32  32 145]
      [ 32  32 145]
      [ 32  32 145]
      ...
      [ 32  32 145]
      [ 32  32 145]
      [ 32  32 145]]
    
     [[ 32  32 145]
      [ 32  32 145]
      [ 32  32 145]
      ...
      [ 32  32 145]
      [ 32  32 145]
      [ 32  32 145]]]


## Convolution


```python
img = cv2.imread('lena.jpg',0)
print(img.shape)

kernel = np.array(([1,0,0],[0,1,0],[0,0,1]),dtype=np.int) 
print(kernel)

output = cv2.filter2D(img,-1,kernel)   #Convolution

cv2.imshow('original',img)
cv2.imshow('Blur',output)
cv2.waitKey()
cv2.destroyAllWindows()

```

    (220, 220)
    [[1 0 0]
     [0 1 0]
     [0 0 1]]


kernel = np.array(([1,0,-1],[2,0,-2],[1,0,-1]),dtype=np.int) #Sobelx (Differentiation in x direction)

kernel = np.array(([1,2,1],[0,0,0],[-1,-2,-1]),dtype=np.int) #Sobely (Differentiation in y direction)

kernel = np.array(([0,-1,0],[-1,5,-1],[0,-1,0]),dtype=np.int) #Sharpen

kernel = np.array([0, 1, 0],[1, -4, 1],[0, 1, 0]), dtype="int") #Laplacian (Double Differentiation)

Try Convolution with the above kernels.


```python
img = cv2.imread('lena.jpg',0)
blur = cv2.GaussianBlur(img,(5,5),0) #Here (5,5) refers to a Gaussian kernel of size 5x5 of zero variance

cv2.imshow('original',img)
cv2.imshow('Blur',blur)
cv2.waitKey()
cv2.destroyAllWindows()

```

### Edge Detection 

We will use the Sobel Operator to detect edges in an image


```python
img = cv2.imread('chessboard.jpg',0)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) 
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

sobel_both = np.sqrt(sobelx**2 + sobely**2 )


cv2.imshow('SobelX',sobelx)
cv2.imshow('SobelY',sobely)
cv2.imshow('Sobel_Both',sobel_both)
cv2.waitKey()
cv2.destroyAllWindows()

```

## Frequency Domain


```python
img = cv2.imread('sine2.jpg',0)
print(img)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

mag_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))  
mag_spectrum = np.asarray(mag_spectrum, dtype = np.uint8)
cv2.imshow('sine',img)
cv2.imshow('Freq-Domain',mag_spectrum)
cv2.waitKey()
cv2.destroyAllWindows()
```

    [[134 147 167 ... 149 164 188]
     [147 160 179 ... 169 183 198]
     [167 179 197 ... 186 200 214]
     ...
     [149 169 186 ... 171 185 204]
     [163 183 200 ... 185 198 216]
     [187 198 214 ... 203 216 228]]


## Morphological Transformations

Erosion and Dilation


```python
img = cv2.imread('Erosion.png',0)
cv2.imshow('Original',img)
cv2.waitKey()
cv2.destroyAllWindows()

ret,th1 = cv2.threshold(img,140,255,cv2.THRESH_BINARY)

kernel = np.ones((5,5),np.uint8)
dilate = cv2.dilate(th1,kernel,iterations = 1)
cv2.imshow('Erosion',erosion)
cv2.imshow('original',th1)
cv2.waitKey()
cv2.destroyAllWindows()

kernel1 = np.ones((5,5),np.uint8)
erosion = cv2.erode(th1,kernel1,iterations = 1)
cv2.imshow('Dilate',dilate)
cv2.imshow('original',th1)
cv2.waitKey()
cv2.destroyAllWindows()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-11-0c77819236e1> in <module>
          8 kernel = np.ones((5,5),np.uint8)
          9 dilate = cv2.dilate(th1,kernel,iterations = 1)
    ---> 10 cv2.imshow('Erosion',erosion)
         11 cv2.imshow('original',th1)
         12 cv2.waitKey()


    NameError: name 'erosion' is not defined


Read the image Sun.jpg and find the position of Sun...


```python
img  = cv2.imread('Sun.jpg',0)
#img = cv2.imread('threshold.jpg',0) #Threshold_adaptive
ret,th1 = cv2.threshold(img,245,255,cv2.THRESH_BINARY)
cv2.imshow('output',th1)
cv2.waitKey()
cv2.destroyAllWindows()

# Write code here
```

### Face Detection

Here we implement the Haar Cascades to detect a face in a given image


```python
import numpy as np
import cv2 as cv
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
while(True):
    im = cv.VideoCapture(0)
    a,img=im.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv.imshow('img',img) 
    cv.waitKey()
    cv.destroyAllWindows()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-224bbfb1915b> in <module>
         13         roi_color = img[y:y+h, x:x+w]
         14         eyes = eye_cascade.detectMultiScale(roi_gray)
    ---> 15     for (ex,ey,ew,eh) in eyes:
         16         cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
         17     cv.imshow('img',img)


    NameError: name 'eyes' is not defined



```python

```


```python

```


```python

```
