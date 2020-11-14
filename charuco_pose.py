import numpy as np
import yaml
import cv2


class CharucoPose:
    def __init__(self, camera_param_path, x, y, square_size, marker_size, marker_ids, is_board):
        self.mtx, self.distCoef = self.load_camera_instrinsics(camera_param_path)
        self.marker_size = marker_size
        self.filter_ids = marker_ids
        
        self.is_board = is_board
        if is_board:
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
            self.board = cv2.aruco.CharucoBoard_create(x, y, square_size, marker_size, aruco_dict) 
               
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
        
        rvecs = rvecs.reshape(1, 1, 3)
        tvecs = tvecs.reshape(1, 1, 3)
        return rvecs, tvecs
    
    def estimate_pose_board(self, ids, corners):
        # Estimate large single marker pose
        if corners is not []:
            rvecs_, tvecs_, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size , self.mtx, self.distCoef)
            retval, rvecs, tvecs = cv2.aruco.estimatePoseBoard(corners, ids, self.board, self.mtx, self.distCoef, rvecs_, tvecs_)
            
            if tvecs is None:
                rvecs = np.nan * np.ones((1, 1, 3))
                tvecs = np.nan * np.ones((1, 1, 3))
                return rvecs, tvecs
        
        rvecs = rvecs.reshape(1, 1, 3)
        tvecs = tvecs.reshape(1, 1, 3)
        return rvecs, tvecs

    def filter_marker_ids(self, corners, ids):
        out_corners = []
        out_ids = []
        if ids is not None:
            for i, id in enumerate(ids):
                for filter_id in self.filter_ids:
                    if id[0] == filter_id:
                        out_corners.append(corners[i])
                        out_ids.append(id)
                        # add 0 id marker only once
                        if filter_id == 0:  
                            return out_corners, np.array(out_ids)


        if len(out_ids) == 0:
            return out_corners, None                        
        return out_corners, np.array(out_ids)

   

    def draw_marker_axis(self, frame, rvec, tvec, length_of_axis):
        return cv2.aruco.drawAxis(frame, self.mtx, self.distCoef, rvec, tvec, length_of_axis)