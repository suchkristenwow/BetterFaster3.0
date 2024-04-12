import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import cv2

'''
detx = {}
detx["clique_id"] = lm_id 
detx["feature_id"] = feat_id
#def get_world_pt(results_dir,experiment,frame,lm_id,width,height,fov,px_coord,camera_pose,depth):
feat_loc = carla_observations_t[lm_id][feat_id]["feat_loc"]
depth = self.get_depth(feat_loc,t)
world_pt = get_world_pt(self.results_dir,self.experiment,t,lm_id,self.width,self.height,self.fov,feat_loc,camera_pose_t,depth)
bearing, range_ = self.get_range_bearing(gt_pose,camera_pose_t)
detx["bearing"] = bearing 
detx["range"] = range_ 
detx["detection"] = 1
observations.append(detx)
'''
def determine_plot_bounds(gt_car_traj):
    #determine plot bounds 
    xmin = min(gt_car_traj[:,0]); xmax = max(gt_car_traj[:,0])
    ymin = min(gt_car_traj[:,1]); ymax = max(gt_car_traj[:,1])

    print("xmin: {},xmax: {}, ymin: {}, ymax: {}".format(xmin,xmax,ymin,ymax))

    x_lower_bound = xmin - np.abs(xmin)*.1; x_upper_bound = xmax + np.abs(xmax)*.1 
    y_lower_bound = ymin - np.abs(ymin)*.1; y_upper_bound = ymax + np.abs(ymax)*.1

    print("x_lower_bound: {}, x_upper_bound: {}".format(x_lower_bound,x_upper_bound))
    print("y_lower_bound: {}, y_upper_bound: {}".format(y_lower_bound,y_upper_bound))

    delta_x = x_upper_bound - x_lower_bound
    delta_y = y_upper_bound - y_lower_bound
    print("delta_x: {},delta_y: {}".format(delta_x,delta_y))

    if max([delta_x,delta_y]) == delta_x:
        print("delta_x is greater...")
        m = np.mean([y_lower_bound,y_upper_bound])
        print("m: ",m)
        y_lower_bound = m - np.abs(delta_x/2); y_upper_bound = m + np.abs(delta_x/2)
    else:
        print("delta_y is greater...")
        m = np.mean([x_lower_bound,x_upper_bound])
        print("m: ",m)
        x_lower_bound = m - np.abs(delta_y/2); x_upper_bound = m + np.abs(delta_y/2)

    print("x_lower_bound: {}, x_upper_bound: {}".format(x_lower_bound,x_upper_bound))
    print("y_lower_bound: {}, y_upper_bound: {}".format(y_lower_bound,y_upper_bound))

    delta_x = x_upper_bound - x_lower_bound
    delta_y = y_upper_bound - y_lower_bound

    print("delta_x: {},delta_y: {}".format(delta_x,delta_y))

    if np.round(delta_x) != np.round(delta_y):
        raise OSError 

    ax_bounds = (x_lower_bound,x_upper_bound,y_lower_bound,y_upper_bound)
    print("ax_bounds: ",ax_bounds)

    return ax_bounds

# Function to update the plot in each animation frame
def update(frame_num, ax, gt_car_traj, ax_bounds):
    ax.clear()
    # Call plot_state function with current frame data
    plot_state(ax, gt_car_traj[frame_num, :], ax_bounds)
    plt.pause(0.01)

def plot_state(ax, robot_pose, ax_bounds):
    ax.clear()
    ax.set_xlim(ax_bounds[0],ax_bounds[1])
    ax.set_ylim(ax_bounds[2],ax_bounds[3])
    ax.set_aspect('equal')

    x = robot_pose[0]
    y = robot_pose[1]
    yaw = robot_pose[5]*(np.pi/180)

    robot_circle = plt.Circle((x, y), 1.5, color='red', fill=True)


    # Calculate the end point of the arrow based on x, y, and yaw
    dx = 3 * np.cos(yaw)
    dy = 3 * np.sin(yaw)
    robot_pointer = ax.arrow(x, y, dx, dy, head_width=2, head_length=2, fc='red', ec='red')

    ax.add_patch(robot_circle)
    ax.add_patch(robot_pointer)
    return ax 


if __name__ == "__main__":
    img = cv2.imread("/media/arpg/easystore/BetterFaster/kitti_carla_simulator/results/depth_images/experiment0_0000_20.png")
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(im_rgb)
    plt.show()