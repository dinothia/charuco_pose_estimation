import matplotlib.pyplot as plt
import numpy as np
import cv2


def interpolate_nan_vals(state):
    state = np.array(state).astype(float)
    ok = np.isnan(state).astype(int) == False
    xp = ok.ravel().nonzero()[0]
    fp = state[np.isnan(state)==False]
    x  = np.isnan(state).ravel().nonzero()[0]

    state[np.isnan(state)] = np.interp(x, xp, fp)
    return state

def rodrigues_to_euler_angles(rvec):
    # Reference: https://www.programcreek.com/python/example/89450/cv2.Rodrigues
    mat, jac = cv2.Rodrigues(rvec)
    sy = np.sqrt(mat[0, 0] * mat[0, 0] + mat[1, 0] * mat[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.math.atan2(mat[2, 1], mat[2, 2])
        y = np.math.atan2(-mat[2, 0], sy)
        z = np.math.atan2(mat[1, 0], mat[0, 0])
    else:
        x = np.math.atan2(-mat[1, 2], mat[1, 1])
        y = np.math.atan2(-mat[2, 0], sy)
        z = 0
    return np.array([x, y, z]) 

def save_pose(filepath, timestamps, tvecs, rvecs):
        with open(filepath, "w") as file:
            for t, tvec, rvec in zip(timestamps, tvecs, rvecs):
                file.write(f"{round(t, 2)}, {tvec[0][0][0]}, {tvec[0][0][1]}, {tvec[0][0][2]}, {rvec[0][0][0]}, {rvec[0][0][1]}, {rvec[0][0][2]}\n")
                
def read_pose(filepath):
    t_list = []
    tvecs_list = []
    euler_list = []

    tvecs = np.zeros((3, len(tvecs_list)))

    with open(filepath, "r") as file:
        for line in file:
            current_line = line.split(",") 
            
            t = np.array(current_line[0]).astype('float')
            tvec = np.array(current_line[1:4]).astype('float')
            rvec = np.array(current_line[4:7]).astype('float')
            euler = np.array(rvec)

            t_list.append(t)
            tvecs_list.append(tvec)
            euler_list.append(euler)
        
        t = np.array(t_list).astype('float')
        tvecs = np.array(tvecs_list).astype('float')
        eulers = np.array(euler_list).astype('float')
    
    return t, tvecs, eulers

def plot_translation(t, tvecs):
    labels = ["x", "y", "z"]

    plt.figure()
    for i in range(3):
        plt.subplot(3, 1, i+1)

        plt.plot(t, interpolate_nan_vals(tvecs[:,i]), label=labels[i] + "-interpolated")        
        plt.plot(t, tvecs[:,i], label=labels[i])        

        plt.xlabel("time [s]")
        plt.ylabel("position [m]")

        plt.xlim([t[0], t[-1]])
        
        plt.legend()

    plt.title("Camera translation")
    plt.tight_layout()

def plot_orientation(t, eulers):
    labels = ["roll", "pitch", "yaw"]

    plt.figure()
    for i in range(3):
        plt.subplot(3, 1, i+1)

        plt.plot(t, interpolate_nan_vals(eulers[:,i]), label=labels[i] + "-interpolated")        
        plt.plot(t, eulers[:,i], label=labels[i])        

        plt.xlabel("time [s]")
        plt.ylabel("angle [deg]")

        plt.xlim([t[0], t[-1]])
        plt.ylim([-180, 180])
        
        plt.legend()

    plt.title("Camera translation")
    plt.tight_layout()

def show_all():
    plt.show()


if __name__ == "__main__":
    filepath = "cameraTrajectory.txt"
    t, tvecs, eulers = read_pose(filepath)
    eulers_deg = 180 * eulers / np.pi

    plot_translation(t, tvecs)
    plot_orientation(t, eulers_deg)
    plt.show()