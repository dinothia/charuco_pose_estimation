import numpy as np
import yaml
import cv2


class CharucoPose:
    def __init__(self, camera_param_path, x, y, square_size, marker_size, marker_id):
        self.mtx, self.distCoef = self.load_camera_instrinsics(camera_param_path)
        self.marker_size = marker_size
        
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
        self.board = cv2.aruco.CharucoBoard_create(x, y, square_size, marker_size, aruco_dict)    
        self.filter_id = marker_id

    def load_camera_instrinsics(self, yaml_path): 
        with open(yaml_path, "r") as file:
            c_p = yaml.load(file, Loader=yaml.FullLoader)
            mtx = np.array([[c_p['fx'], 0, c_p['cx']], [0, c_p['fy'], c_p['cy']],[0,0,1]])
            distCoef = np.array([c_p['k1'], c_p['k2'], c_p['p1'], c_p['p2']])
            return mtx, distCoef
        return None, None

    def load_marker_params(self, x, y, square_size, marker_size, aruco_dict):
        """
        x : number of squares along x-axis
        y : number of squares along y-axis
        square_size : in meters
        marker_size : in meters
        """
        board = cv2.aruco.CharucoBoard_create(x, y, square_size, marker_size, aruco_dict)    
        return board

    def estimate_pose(self, corner):
        # Estimate large single marker pose
        if corner is not []:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corner, self.marker_size , self.mtx, self.distCoef)
            
            if tvecs is None:
                rvecs = np.nan * np.ones((1, 1, 3))
                tvecs = np.nan * np.ones((1, 1, 3))
            return rvecs, tvecs
        
        rvecs = np.nan * np.ones((1, 1, 3))
        tvecs = np.nan * np.ones((1, 1, 3))
        return rvecs, tvecs
    
    def filter_ids(self, ids, corners):
        if ids is not None:
            for i, id in enumerate(ids):
                if id == self.filter_id:
                    return np.array([id]), [corners[i]]
        return None, []
