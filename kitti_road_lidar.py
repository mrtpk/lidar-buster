import numpy as np
import copy

class kittiRoad(object):
    '''
    Collection of helpers for processing LiDAR point cloud provided in KITTI road evaluation dataset.
    '''
    def read_kitti_calib(self, file_path):
        '''
        Reads calib file of KITTI's road evaluation dataset
        '''
        float_chars = set("0123456789.e+- ")
        calib_data = {}
        with open(file_path, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                value = value.strip()
                calib_data[key] = value
                if float_chars.issuperset(value):
                        # each value is stored in float64
                        calib_data[key] = np.array(list(map(float, value.split(' '))))
        return calib_data
    
    def get_projection_mappings_selector(self, pixel_mappings, img):
        '''
        Returns a boolean mask for pixel mappings inside the image
        '''
        x, y = pixel_mappings
        h, w = img.shape[0], img.shape[1]
        return (y < h) * (y > 0) * (x < w) * (x > 0)
    
    def get_projection_mappings(self, points, calib_data, img=None, return_selector=False):
        '''
        Returns pixel mappings for each point in the LiDAR point cloud
        for projecting it onto an image.
        '''
        transformation = calib_data['Tr_velo_to_cam'].reshape(3, 4)
        rectification = calib_data['R0_rect'].reshape(3, 3)
        projection = calib_data['P2'].reshape(3, 4)

        _points = copy.deepcopy(points.T)

        # Convert filtered velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c) 
        for i in range(_points.shape[1]):
            _points[:3,i] = np.matmul(transformation, _points[:,i]) # only x, y, z are changed.

        # Rectification
        _points = np.delete(_points, 3, axis=0) # we don't need r amymore
        for i in range(_points.shape[1]):
            _points[:,i] = np.matmul(rectification, _points[:,i])

        # Convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y)
        projection = projection[:3, :3]
        for i in range(_points.shape[1]):
            _points[:,i] = np.matmul(projection, _points[:,i]) 
        # Normalize
        _points = _points[::]/_points[::][2]
        _points = np.delete(_points, 2, axis=0)
        _points = np.floor(_points).astype(int)
        
        x, y = _points
        if img is None:
            return x, y
        selector = self.get_projection_mappings_selector(pixel_mappings=_points, img=img)
        if return_selector:
            return x[selector], y[selector], selector
        return x[selector], y[selector]
