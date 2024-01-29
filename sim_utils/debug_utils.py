import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os 
import numpy as np

def build_projection_matrix(w, h, fov):
    #focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    focal = w / (2*np.tan(fov))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def compute_yaw_pitch(point1, point2):
    # compute the direction vector from point1 to point2
    direction_vector = point2 - point1
    
    # compute the relative yaw, pitch, and roll
    yaw = np.arctan2(direction_vector[1], direction_vector[0])
    pitch = np.arctan2(direction_vector[2], np.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2))
    return yaw,pitch

def parse_carla_path(results_dir,run,sim_length):
    traj_file = os.path.join(results_dir,"experiment" + str(run) + "_gt_car_pose.txt")
    exp_traj = []
    with open(traj_file) as f:
        lines = f.readlines()
        for line in lines:
            comma_idx = [i for i,x in enumerate(line) if x ==","]
            x_idx0 = line.index("x=")
            y_idx0 = line.index("y=")
            roll_idx = line.index("roll=")
            pitch_idx = line.index("pitch=")
            yaw_idx0 = line.index("yaw=")
            x = float(line[x_idx0+2:comma_idx[0]])
            y = float(line[y_idx0+2:comma_idx[1]])
            roll = float(line[roll_idx+5:-3])
            pitch = float(line[pitch_idx+6:comma_idx[3]])
            yaw = float(line[yaw_idx0+4:comma_idx[-1]])
            exp_traj.append([x,y,roll,pitch,yaw])
    #downsample the traj measurements according to the sim length start:stop:step
    step = int(len(exp_traj) / sim_length)
    downsample_exp_traj = exp_traj[0:len(exp_traj):step]
    downsample_exp_traj = np.reshape(downsample_exp_traj,(sim_length,5))
    downsample_exp_traj[:,2] = np.deg2rad(downsample_exp_traj[:,2])
    return downsample_exp_traj

def car2_cam_trans(robot_disp,camera_offset):
    cam_x = robot_disp[0] + camera_offset[0]*np.cos(robot_disp[-1]) + camera_offset[1]*np.sin(robot_disp[-1])
    cam_y = robot_disp[1] + camera_offset[0]*np.sin(robot_disp[-1]) + camera_offset[1]*np.cos(robot_disp[-1])
    cam_z = 1.7
    camera_location = [cam_x,cam_y,cam_z]
    return camera_location

def get_image_px_bbs(img_dir,obj_file_name,frame_str="0000",experiment=0):
    img = cv2.imread(os.path.join(img_dir,"experiment"+str(experiment)+"_"+frame_str+"_0.png"))
    h,w,c = img.shape
    print("img.shape: ",img.shape)
    h_fov = np.pi/2; v_fov = (h/w) * np.pi/2
    print("h_fov: {} v_fov: {}".format(h_fov,v_fov))
    #cam_trans = np.genfromtxt("/media/kristen/easystore2/run_carla_experiments/empty_map_results/camera_transformations/experiment" + str(experiment) +"_frame"+ frame_str + ".csv",delimiter=",")
    object_3d_world_pts = np.genfromtxt(os.path.join("/media/kristen/easystore2/run_carla_experiments/empty_map_results/lm_pts/",obj_file_name),delimiter=",")
    car_traj = parse_carla_path("/media/kristen/easystore2/run_carla_experiments/empty_map_results/gt_car_poses",experiment,500)
    three_d_car_pose = car_traj[0,:]; 
    print("three_d_car_pose: ",three_d_car_pose)
    camera_offset = [0.30,0,1.7]
    cam_location = car2_cam_trans(three_d_car_pose,camera_offset)
    bb_px_coords = []
    for pt in object_3d_world_pts:
        yaw,pitch = compute_yaw_pitch(cam_location,pt)
        print("yaw: {} pitch: {}".format(yaw,pitch))
        
        x_pixel = np.round((w / h_fov)*yaw + (w / 2))
        y_pixel = np.round((h / v_fov)*pitch + (h/2))

        if not 0<= x_pixel <= w:
            print("x_pixel: {} y_pixel: {}".format(x_pixel,y_pixel))
            raise OSError
        if not 0 <= y_pixel <= h:
            print("x_pixel: {} y_pixel: {}".format(x_pixel,y_pixel))
            raise OSError
        
        bb_px_coords.append([x_pixel,y_pixel])

    return bb_px_coords

def plot_orb_feats(input_dir,orb_dir,test_file_name="experiment0_0000"):

    test_file = os.path.join(input_dir,test_file_name + "_0.png")
    img = cv2.imread(test_file)
    # create a figure and a set of subplots
    fig, ax = plt.subplots()
    # display the image
    ax.imshow(img)
    #plot the pixels 
    orb_pts = np.genfromtxt(os.path.join(orb_dir,test_file_name + ".csv"),delimiter=",")
    #print("orb_pts.shape: ",orb_pts.shape)
    ax.scatter(orb_pts[:,0],orb_pts[:,1],c="red",marker="o",s=6)
    plt.show()

def parse_gt_lm_data(results_dir,experiment,type):
    if "cone" in type:
        sub_dir = "gt_cone_data"
    else:
        sub_dir = "gt_tree_data"
    file = os.path.join(results_dir,sub_dir,"experiment"+str(experiment)+"_gt_cone_data.csv")
    data = np.genfromtxt(file,delimiter="")
    data = data[1:] #the first entry in the row is the experiment number 
    n_lms = int(len(data) / 4)
    print("There are {} {}s".format(n_lms,type))
    lm_gt_locations = []
    for i in range(n_lms):
        lm_x = data[i*4 + 1]
        lm_y = data[i*4 + 2]
        lm_z = data[i*4 + 3]
        lm_gt_locations.append([lm_x,lm_y,lm_z])

    return np.array(lm_gt_locations)


def wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

if __name__ == "__main__":
    results_dir = "/media/kristen/easystore2/BetterFaster2.0/run_carla_experiments/empty_map_results"
    orb_dir = "/media/kristen/easystore2/ORB_feat_extraction/empty_map_results_orb_feats"
    img_dir = "/media/kristen/easystore2/BetterFaster2.0/run_carla_experiments/empty_map_results/rgb_images"
    lm_pt_dir = "/media/kristen/easystore2/BetterFaster2.0/run_carla_experiments/empty_map_results/lm_2dpts"
    
    #pick the first file
    #file_name = [x for x in os.listdir(lm_pt_dir) if x[-3:] == "csv" if "experiment0" in x][0]
    #print("file_name: ",file_name)
    #first_file = os.path.join(lm_pt_dir,file_name)
    first_file = os.path.join(lm_pt_dir,"experiment0frame0000tree2127.csv")
    #x_min,x_max,y_min,y_max
    lm_bb_coords = np.genfromtxt(first_file,delimiter=",")
    x_min = lm_bb_coords[0]; x_max = lm_bb_coords[1]
    y_min = lm_bb_coords[2]; y_max = lm_bb_coords[3]

    base_dir = "/media/kristen/easystore2/ORB_feat_extraction"
    orb_pts = np.genfromtxt(os.path.join(base_dir,"example.csv"),delimiter=",")
    print("min(orb_pts[:,0]):{} min(orb_pts[:,1]):{}".format(min(orb_pts[:,0]),min(orb_pts[:,1])))
    print("max(orb_pts[:,0]):{} max(orb_pts[:,1]):{}".format(max(orb_pts[:,0]),max(orb_pts[:,1])))
    
    print("x_min: {} x_max: {} y_min: {} y_max: {}".format(x_min,x_max,y_min,y_max))
    idx_x = [i for i,x in enumerate(orb_pts[:,0]) if x_min <  x < x_max]
    print("len(idx_x):",len(idx_x))
    idx_y = [i for i,x in enumerate(orb_pts[:,1]) if y_min < x < y_max]
    print("len(idx_y):",len(idx_y))
    idx = np.array([x for x in idx_x if x in idx_y])
    print("idx: ",idx)
    

    #img = cv2.imread(os.path.join(img_dir,file_name[:-3] + "png"))
    #print("file: ",os.path.join(img_dir,file_name[:-3] + "png"))

    img = cv2.imread(os.path.join(os.path.join(lm_pt_dir,"camera"),"experiment0frame0000tree2127.png"))
    
    cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,0,255, 255), 1)
    cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
    cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,0,255, 255), 1)
    cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,0,255, 255), 1)

    fig, ax = plt.subplots()
    # display the image
    ax.imshow(img)
    
    ax.scatter(orb_pts[:,0],orb_pts[:,1],c="red",marker="o",s=6)
    if len(idx) > 0:
        ax.scatter(orb_pts[idx,0],orb_pts[idx,1],c="green",marker="o",s=6)
    
    plt.show()