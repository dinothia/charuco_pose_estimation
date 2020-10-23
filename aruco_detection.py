import cv2


class ArucoDetector:
    def __init__(self):
        self.dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)

        # Detection parameters
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.parameters.adaptiveThreshWinSizeMin        = 3
        self.parameters.adaptiveThreshWinSizeStep       = 1
        self.parameters.adaptiveThreshConstant          = 5
        self.parameters.minMarkerPerimeterRate          = 0.005
        self.parameters.cornerRefinementMethod          = cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize         = 3
        self.parameters.cornerRefinementMaxIterations   = 50
        self.parameters.cornerRefinementMinAccuracy     = 0.05
        #parameters.polygonalApproxAccuracyRate          = 0.1
        #parameters.adaptiveThreshWinSizeMax             = 50

    def get_corner_and_ids(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.corners, self.ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, self.dict, parameters=self.parameters)
        return self.corners, self.ids

    def draw_markers(self, frame, corners, ids):
        return cv2.aruco.drawDetectedMarkers(frame, corners, ids)#, borderColor=(100, 0, 240))