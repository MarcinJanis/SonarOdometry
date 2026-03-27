import numpy as np 
import cv2

# =============================================================== #
# Function for remaping between polar and carthesian coord system #
# =============================================================== #

'''
     Forward mapping (theta, r) -> (x, y)

1. recenter and set correct orientation (0, w), (0, h) -> (-w/2, w/2), (h, 0)
2. rescale to real world value
3. (x, y) = r * (sin(theta) cos(theta)) 
4. rescale to image value


===> Inverse mapping (x, y) -> (theta, r) <===

1. rescale to real world value
2. r = sprt(x^2 + y^2), theta = atan2(x/y)
3. rescale to img value
4. reverse orientation and range to (0, w) and (0, h)
'''

def img_polar2cart(I, r_min, r_max, theta_max, out_shape = None, bg = 0):

  h, w = I.shape
  
  if out_shape is None: 
    out_shape = (h, 2*h)
    
  # inverse mapping 
  # for (x, y) on target img, find coords of (theta, r) on orginal img
  x, y = np.meshgrid(np.arange(out_shape[1]), np.arange(out_shape[0]))

  # recenter
  # x (0, w) and y (0, h) -> x (- w/2 , w/2) and y (h, 0) - correct orientation 
  x = x - out_shape[1]/2
  y = out_shape[0] - y
  
  # rescale to real-world values
  scale = (r_max - r_min) / out_shape[0]
  x_r = x * scale 
  y_r = y * scale + r_min

  # create mapping
  # (x, y) -> (theta, r)
  r = np.sqrt(x_r**2 + y_r**2)
  y_r_clamp = np.maximum(y_r, 1e-5) 
  theta = np.arctan2(x_r, y_r_clamp)

  # rescale to orginal 0 - 1 range
  r = (r - r_min) / (r_max - r_min) * h
  theta = (theta / theta_max) * w 

  # recenter
  r = h - r
  theta = theta + w / 2

  
  out_img = cv2.remap(I, theta.astype(np.float32), r.astype(np.float32), 
                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bg)


  return out_img


def img_cart2polar(I, r_min, r_max, theta_max, out_shape = None, bg = 0):

  h, w = I.shape
  
  if out_shape is None: 
    out_shape = (h, h//2)
    
  # inverse mapping 
  # for (theta, r) on target img, find coords (x, y) on orginal img
  
  theta, r = np.meshgrid(np.arange(out_shape[1]), np.arange(out_shape[0]))

  # recenter
  # x (0, w) and y (0, h) -> x (- w/2 , w/2) and y (h, 0) - correct orientation 
  theta = theta - out_shape[1]/2
  r = out_shape[0] - r
  
  # rescale to real-world values

  theta = theta / out_shape[1] * theta_max
  r = r / out_shape[0] * (r_max - r_min)  + r_min

  # create mapping
  # (theta, r) - > (x, y)
  x = r * np.sin(theta)
  y = r * np.cos(theta)

  # rescale to input image size 
  scale = h / (r_max - r_min) 
  x = x * scale 
  y = (y - r_min) * scale
  
  # recenter
  x = x + w / 2
  y = h - y

  
  out_img = cv2.remap(I, x.astype(np.float32), y.astype(np.float32), 
                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bg)


  return out_img




# =============================================================== #
# Class for faster remapping video stream                         #
# Create maps once, and keep in RAM                               #
# =============================================================== #


class fast_polar_cart_transform:
  def __init__(self, r_min, r_max, theta_max, input_shape, out_shape = None, bg = 0):

    self.bg = bg

    self.input_shape = input_shape
    
    if out_shape is None: 
      self.out_shape = (h, 2*h)
    else:
      self.out_shape = out_shape

    self.r_min = r_min
    self.r_max = r_max
    self.theta_max = theta_max 
    

    def create_polar2cart_remap():
          
      h, w = self.input_shape
      
      if out_shape is None: 
        out_shape = (h, 2*h)
        
      # inverse mapping 
      # for (x, y) on target img, find coords of (theta, r) on orginal img
      x, y = np.meshgrid(np.arange(out_shape[1]), np.arange(out_shape[0]))
    
      # recenter
      # x (0, w) and y (0, h) -> x (- w/2 , w/2) and y (h, 0) - correct orientation 
      x = x - out_shape[1]/2
      y = out_shape[0] - y
      
      # rescale to real-world values
      scale = (r_max - r_min) / out_shape[0]
      x_r = x * scale 
      y_r = y * scale + r_min
    
      # create mapping
      # (x, y) -> (theta, r)
      r = np.sqrt(x_r**2 + y_r**2)
      y_r_clamp = np.maximum(y_r, 1e-5) 
      theta = np.arctan2(x_r, y_r_clamp)
    
      # rescale to orginal 0 - 1 range
      r = (r - r_min) / (r_max - r_min) * h
      theta = (theta / theta_max) * w 
    
      # recenter
      r = h - r
      theta = theta + w / 2

      # save remap
      self.x_map = theta.astype(np.float32)
      self.y_map = r.astype(np.float32

    def create_cart2polar_remapping():
      
      h, w = self.input_shape
      
      if out_shape is None: 
        out_shape = (h, h//2)
        
      # inverse mapping 
      # for (theta, r) on target img, find coords (x, y) on orginal img
      
      theta, r = np.meshgrid(np.arange(out_shape[1]), np.arange(out_shape[0]))
    
      # recenter
      # x (0, w) and y (0, h) -> x (- w/2 , w/2) and y (h, 0) - correct orientation 
      theta = theta - out_shape[1]/2
      r = out_shape[0] - r
      
      # rescale to real-world values
    
      theta = theta / out_shape[1] * theta_max
      r = r / out_shape[0] * (r_max - r_min)  + r_min
    
      # create mapping
      # (theta, r) - > (x, y)
      x = r * np.sin(theta)
      y = r * np.cos(theta)
    
      # rescale to input image size 
      scale = h / (r_max - r_min) 
      x = x * scale 
      y = (y - r_min) * scale
      
      # recenter
      x = x + w / 2
      y = h - y

      # save remap
      self.x_map = x.astype(np.float32)
      self.y_map = y.astype(np.float32)

      

                            
    def __call__(self, img):
      
      out_img = cv2.remap(img, self.x_map, self.y_map, 
                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.bg)
      
      return out_img
    
