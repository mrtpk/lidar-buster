import numpy as np
class LidarTools(object):
    '''
    Collection of helpers for processing LiDAR point cloud.
    '''
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
    
    def filter_points(self, points, side_range=None, fwd_range=None, \
                  height_range=None, horizontal_fov=None, vertical_fov=None):
        '''
        Returns filtered points based on side, forward and height range, and, horizontal and vertical field of view.
        '''
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        r = points[:, 3]
        
        mask = np.full_like(x, True)
        
        if side_range is not None:
            side_mask = np.logical_and((y > -side_range[1]), (y < -side_range[0]))
            mask = np.logical_and(mask, side_mask)

        if fwd_range is not None:
            fwd_mask = np.logical_and((x > fwd_range[0]), (x < fwd_range[1]))
            mask = np.logical_and(mask, fwd_mask)

        if height_range is not None:
            height_mask = np.logical_and((z > height_range[0]), (z < height_range[1]))
            mask = np.logical_and(mask, height_mask)
            
        if horizontal_fov is not None:
            horizontal_fov_mask = np.logical_and(np.arctan2(y, x) > (-horizontal_fov[1] * np.pi / 180), \
                            np.arctan2(y, x) < (-horizontal_fov[0] * np.pi / 180))
            mask = np.logical_and(mask, horizontal_fov_mask)
        
        if vertical_fov is not None:
            distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            vertical_fov_mask = np.logical_and(np.arctan2(z,distance) < (vertical_fov[1] * np.pi / 180), \
                            np.arctan2(z,distance) > (vertical_fov[0] * np.pi / 180))
            mask = np.logical_and(mask, vertical_fov_mask)

        indices = np.argwhere(mask).flatten()
        return points[indices, :]    