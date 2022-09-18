class WhiteBalanceTool():
    
    def __init__(self, brighness_mode=0):
        """
        Temp tool to estimate and change RGB image temperature
        
        Methods:
        --------
        1. `get_temperature`: estimates the current temperature of an image:
            parameters:
                - img: tf.tensor([h,w,3], tf.float32) --> Input Image
            returns:
                - temp_estimate: tf.tensor([1], tf.int32) --> Temperature Estimation in Kelvin
        
        2. `convert_temperature`: to change the temperature of an image
            parameters:
                - img: tf.tensor([h,w,3], tf.float32) --> Input Image
            returns:
                - temp_image: tf.tensor([h,w,3], tf.float32) --> Input image at *new* temperature
                
        Theory:
        -------
        A. Temperature Estimation is done by:
            1. Find the white-point of the input image (brightest pixel)
            2. Find the closest (Sum MAE) RGB value in the temperature table
            3. Take temperature of closest RGB to white-point
            
        B. Temperature conversion is done by:
            1. Find the white-point of the input image (brightest pixel)
            2. Look-up RGB value of requested temperature
            3. Scale this RGB value by the white-point RGB values
            4. Apply this weight to the input img RGB channels
        """
        if brighness_mode==0:
            self.get_brightness_image = self.get_brightness_image_v1
        elif brighness_mode==1:
            self.get_brightness_image = self.get_brightness_image_v2
        else:
            self.get_brightness_image = self.get_brightness_image_v3

    #########################################
    ###  A. ESTIMATING IMAGE TEMPERATURE  ###
    #########################################
    

    @tf.function
    def get_brightness_image_v1(self, img):
        """Returns the brightness of each pixel in the given image
        https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
        """
        # img is [h,w,3] tensor
        
        # To compute brightness of an image do a weighted average across channels
        brightness_image = tf.convert_to_tensor([0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]], dtype=tf.float32)
        # brightness_image will be a [1,h,w] tensor
    
        return brightness_image 
    
    
    @tf.function
    def get_brightness_image_v2(self, img):
        """Returns the brightness of each pixel in the given image
        https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
        """
        #
        # img is [h,w,3] tensor
        
        # To compute brightness of an image do a weighted average across channels
        brightness_image = tf.convert_to_tensor([0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]], dtype=tf.float32)
        # brightness_image will be a [1,h,w] tensor
    
        return brightness_image 
    
    
    @tf.function
    def get_brightness_image_v3(self, img):
        """Returns the brightness of each pixel in the given image
        https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
        """
        # img is [h,w,3] tensor
        
        # To compute brightness of an image do a weighted average across channels
        brightness_image = tf.convert_to_tensor([0.299*img[:,:,0]**2 + 0.587*img[:,:,1]**2 + 0.114*img[:,:,2]**2], dtype=tf.float32)
        # brightness_image will be a [1,h,w] tensor
    
        return brightness_image 
    
    
    @tf.function
    def get_white_point(self, img):
        """Returns white point and its coordinates"""
        # img is a [h,w,3] tensor 
        brightness_image = self.get_brightness_image(img) 
        #brightness_image is [1,h,w] tensor
        
        # get the maximum pixel (from all channels collectively)
        # brightness_image[0] is [h,w] tensor
        brightness_argmax = tf.argmax(tf.reshape(brightness_image[0], [-1]), axis=None) #brightness_argmax is () location
        
        brightness_argmax = brightness_argmax // brightness_image[0].shape[1], \
                            brightness_argmax % brightness_image[0].shape[1],
        # brightness_argmax is tuple of height and width of argmax
        
        # white_point = [y,x,:]  3-D array of (R,G,B) channel values of highest brightness, where
        # brightness(y,x) = 0.2126*img[y,x,0] + 0.7152*img[y,x,1] + 0.0722*img[y,x,2] 
        white_point = tf.gather_nd(img, brightness_argmax)
        
        return white_point, brightness_argmax
        

    #########################################
    ###  B. CONVERTING IMAGE TEMPERATURE  ###
    #########################################
    
    @tf.function
    def convert_temperature(self, img):
        """After estimate white point of img as the pixel with maximum brightness cast the image to another temp and return"""
        # img is [h,w,3] tensor
        
        #Ensure RGB or RGBA float32 tensor - alpha channel is dropped
        assert ((tf.is_tensor(img)) & (len(img.shape) ==3) & (img.shape[-1] in [3,4])), "input should be 4 channel RGB float32 tensor"
        
        img = tf.cast(img[:, :, :3], dtype=tf.float32)
        
        #Get brightess pixel location and its RGB value
        white_point, white_point_idx = self.get_white_point(img) # e.g. white_point = [248, 255, 234]
        
        # how much to scale the values of R, G and B channels
        matrix = tf.convert_to_tensor([
            [255.0/white_point[0], 0.0, 0.0],
            [0.0,255.0/white_point[1], 0.0],
            [0.0, 0.0, 255.0/white_point[2]],
        ])
        
        img_out = tf.matmul(img, matrix)
        
        return tf.cast(tf.clip_by_value(img_out, 0.0, 255.0), tf.uint8)