import numpy as np
class LidarTools(object):
    def get_bev(self, points, resolution=0.1, pixel_values=None, generate_img=None):  
        '''
        Returns bird's eye view of a LiDAR point cloud for a given resolution.
        Optional pixel_values can be used for giving color coded info the point cloud.
        Optional generate_img function can be used for creating images.
        '''
        
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        x_range = -1 * np.ceil(y.max()).astype(np.int), ((y.min()/np.abs(y.min())) * np.floor(y.min())).astype(np.int)
        y_range = np.floor(x.min()).astype(np.int), np.ceil(x.max()).astype(np.int)

        # Create mapping from a 3D point to a pixel based on resolution
        # floor() used to prevent issues with -ve vals rounding upwards causing index out bound error
        x_img = (-y / resolution).astype(np.int32) - int(np.floor(x_range[0]/resolution))
        y_img = (x / resolution).astype(np.int32) - int(np.floor(y_range[0]/resolution))

        img_width  = int((x_range[1] - x_range[0])/resolution)
        img_height = int((y_range[1] - y_range[0])/resolution)

        if pixel_values is None:
            pixel_values = (((z - z.min()) / float(z.max() - z.min())) * 255).astype(np.uint8)

        if generate_img is None:
            img = np.zeros([img_height, img_width], dtype=np.uint8)
            img[-y_img, x_img] = pixel_values
            return img
        
        return generate_img(img_height, img_width, -y_img, x_img, pixel_values)    