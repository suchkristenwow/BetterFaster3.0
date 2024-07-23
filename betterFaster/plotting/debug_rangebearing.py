import numpy as np 
import os 
import toml 
import matplotlib.pyplot as plt 
import math 
import pickle 
import cv2 
from scipy.spatial.transform import Rotation  
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline 

class CamProjector:
    def __init__(self,K,camera_pose,depth) -> None:
        #self.cam_model = CamProjector.get_camera_model(K,R,P,D,width,height)
        self.camera_pose = camera_pose
        #self.robot_pose = robot_pose #[x,y,z,roll,pitch,yaw]
        self.depth = depth
        self.K = K 

    @staticmethod
    def projectPixelTo3dRay(cx,cy,fx,fy,uv):
        """
        :param uv:        rectified pixel coordinates
        :type uv:         (u, v)

        Returns the unit vector which passes from the camera center to through rectified pixel (u, v),
        using the camera :math:`P` matrix.
        This is the inverse of :meth:`project3dToPixel`.
        """
        x = (uv[0] - cx) / fx
        y = (uv[1] - cy) / fy
        norm = math.sqrt(x*x + y*y + 1)
        x /= norm
        y /= norm
        z = 1.0 / norm
        return np.array([x,y,z])


    @staticmethod
    def pose_to_transformation_matrix(pose):
        tf_matrix = np.zeros((4,4))
        p = [pose[3],pose[4],pose[5]]
        r = Rotation.from_euler("XYZ", p, degrees=False)
        tf_matrix[:3,:3] = r.as_matrix()
        tf_matrix[0,3] = pose[0]
        tf_matrix[1,3] = pose[1]
        tf_matrix[3,3] = pose[2]
        tf_matrix[3,3] = 1
        return tf_matrix

    def project_pixel(self, px):
        #ray = np.asarray(self.cam_model.projectPixelTo3dRay(pixel))
        cx = self.K[0,2]; cy = self.K[1,2]; fx = self.K[0,0]; fy = self.K[1,1]
        ray = self.projectPixelTo3dRay(cx,cy,fx,fy,px)
        # Convert to Point
        point = ray * self.depth
        return point
    
    def convert_optical_to_nav(self, cam_point):
        '''
        cam_nav_frame_point = Point()
        cam_nav_frame_point.x = cam_point[2]
        cam_nav_frame_point.y = -1.0 *cam_point[0]
        cam_nav_frame_point.z = -1.0 * cam_point[1]
        '''
        #cam_nav_frame_point = (cam_point[0],cam_point[2],cam_point[1])
        cam_nav_frame_point = (cam_point[2],-1*cam_point[0],-1*cam_point[1])
        return cam_nav_frame_point
    
    def apply_cam_transformation(self, point):
        full_tf = self.camera_pose 
        point_np = np.array(point); point_np = np.append(point,1)
        new_point = np.dot(full_tf, point_np)
        return new_point
    
    def project(self,pixel):
        # Project To A Point in Camera frame
        cam_point = self.project_pixel(pixel)
        # Camera Point To Cam Nav Frame
        cam_point_frame = self.convert_optical_to_nav(cam_point)
        # Transfrom point
        new_point = self.apply_cam_transformation(cam_point_frame)
        return new_point
    
def get_range_bearing(p0,p1,sensor_noise_variance,enable_noise_variance=True):
    '''
    if len(robot_pose) == 6:
        robot_heading = robot_pose[5]
    elif len(robot_pose) == 3: 
        robot_heading = robot_pose[2]
    print("robot_heading: ",robot_heading)
    '''
    # Extract x, y coordinates from p0 and p1
    x0, y0 = p0[0], p0[1] #camera pose
    if enable_noise_variance:
        x1, y1 = p1[0], p1[1] #landmark pos
        #print("p1: ",p1)
        noisy_landmark = p1 + np.random.normal(0,sensor_noise_variance,size=3)
        x1 = noisy_landmark[0]; y1 = noisy_landmark[1]
    else:
        x1, y1 = p1[0], p1[1] #landmark pos            

    #print("This is the camera pose - x0: {}, y0: {}".format(x0,y0))
    #print("This is the landmark pos - x1: {},y1:{}".format(x1,y1))
    # Calculate the difference in x and y coordinates
    dx = x1 - x0
    dy = y1 - y0

    #print("dx: {},dy: {}".format(dx,dy))
    # Calculate the range between the two points
    #range_ = math.sqrt(dx**2 + dy**2)
    range_ = np.linalg.norm(p0[:2] - p1[:2])
    #print("range: ",range_)

    # Calculate the relative yaw (angle) from p0 to p1
    relative_yaw = math.atan2(dy, dx)
    #print("relative_yaw: ",math.degrees(relative_yaw))

    #print("relative_yaw: {}, robot_heading: {}".format(math.degrees(relative_yaw),robot_heading))
    #yaw = math.degrees(relative_yaw) + robot_heading
    yaw = math.degrees(relative_yaw)
    #print("this is yaw:",yaw)

    test_0 = x0 + range_*np.cos(np.deg2rad(yaw))
    test_1 = y0 + range_*np.sin(np.deg2rad(yaw))

    if int(np.round(x1,1) - np.round(test_0,1)) > 0:
        #print("np.round(test_0,1):{}, np.round(x1,1):{}".format(np.round(test_0,1),np.round(x1,1)))
        delta_x = np.round(x1,1) - np.round(test_0,1) 
        #print("delta_x: ",delta_x)
        #print("test_0:{},test_1: {}".format(test_0,test_1))
        fig,ax = plt.subplots()
        ax.scatter(x0,y0,color="k")
        pointer_x = x0 + 2*np.cos(np.deg2rad(p0[5]))
        pointer_y = y0 + 2*np.sin(np.deg2rad(p0[5]))
        ax.plot([x0,pointer_x],[y0,pointer_y],"k")
        ax.scatter(x1,y1,color="r",marker="*")
        ax.scatter(test_0,test_1,color="b",marker="*")
        ax.set_aspect('equal')
        plt.show(block=True)
        raise OSError 

    if int(np.round(y1,1) - np.round(test_1,1)) > 0: 
        #print("np.round(test_0,1):{}, np.round(x1,1):{}".format(np.round(test_0,1),np.round(x1,1)))
        delta_y = np.round(test_0,1) - np.round(x1,1)
        #print("delta_y: ",delta_y)
        #print("test_0:{},test_1: {}".format(test_0,test_1))
        fig,ax = plt.subplots()
        ax.scatter(x0,y0,color="k")
        pointer_x = x0 + 2*np.cos(np.deg2rad(p0[5]))
        pointer_y = y0 + 2*np.sin(np.deg2rad(p0[5]))
        ax.plot([x0,pointer_x],[y0,pointer_y],"k")
        ax.scatter(x1,y1,color="r",marker="*")
        ax.scatter(test_0,test_1,color="b",marker="*")
        ax.set_aspect('equal')
        plt.show(block=True)
        raise OSError 

    return yaw, range_

def get_depth(exp,results_dir,frame,px_coord,px_radius=3):
    depth_image = results_dir + "/depth_images/experiment" + str(exp) +"_"+ str(frame).zfill(4) + "_20.png"
    if not os.path.exists(depth_image): 
        print("depth_image: ",depth_image)
        raise OSError
    im = cv2.imread(depth_image)
    x = int(px_coord[0]); y = int(px_coord[1])
    C = im[y][x]
    B = C[0]; G = C[1]; R = C[2]
    normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)

    if normalized * 1000 > 900:
        #try searching a 2 px radius 
        min_x_coord = int(px_coord[0]) - px_radius if 0 <= int(px_coord[0]) - px_radius else 0 
        max_x_coord = int(px_coord[0]) + px_radius if int(px_coord[0]) + px_radius <= im.shape[1] - 1 else im.shape[1] - 1 
        min_y_coord = int(px_coord[1]) - px_radius if 0 <= int(px_coord[1]) - px_radius else 0 
        max_y_coord = int(px_coord[1]) + px_radius if int(px_coord[1]) + px_radius <= im.shape[0] - 1 else im.shape[0] - 1 
        potential_depths = []
        for x_coord in range(min_x_coord,max_x_coord):
            for y_coord in range(min_y_coord,max_y_coord): 
                C = im[y_coord][x_coord]
                B = C[0]; G = C[1]; R = C[2]
                pot_depth = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
                potential_depths.append(pot_depth)
        normalized = min(potential_depths)

    return normalized * 1000

def build_projection_matrix(fov,width,height):
    #if not isinstance(fov,int):
    if fov < 2*np.pi:
        #fov is probably in radians but we were expecting degrees
        fov = fov * (180/np.pi)
    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    v_focal = height / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = focal 
    K[1, 1] = v_focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    return K

def get_world_pt(fov,width,height,camera_pose,depth,px):
    K = build_projection_matrix(fov,width,height)
    cam_projector = CamProjector(K,camera_pose,depth)
    world_pt = cam_projector.project(px)
    return world_pt[:3]

def get_camera_pose(results_dir,exp,frame):
    #file_path = os.path.join(self.results_dir,"camera_transformations/experiment"+str(self.experiment)+"_frame"+str(frame).zfill(4)+".csv")
    file_path = os.path.join(results_dir,"depth_images/transformations/experiment"+str(exp)+ "_" + str(frame+1).zfill(4)+"_20.csv")
    if not os.path.exists(file_path):
        print("file_path: ",file_path)
        raise OSError 
    '''
    #get trans matrix 
    inv_trans_matrix = np.genfromtxt(file_path,delimiter=",")
    trans_matrix = np.linalg.inv(inv_trans_matrix)

    # Extract translation vector from the transformation trans_matrix
    translation = trans_matrix[:3, 3]

    # Extract rotation trans_matrix from the transformation trans_matrix
    rotation_trans_matrix = trans_matrix[:3, :3]

    # Extract roll, pitch, and yaw angles from the rotation trans_matrix
    # Roll (rotation about x-axis)
    roll = math.atan2(rotation_trans_matrix[2, 1], rotation_trans_matrix[2, 2])
    # Pitch (rotation about y-axis)
    pitch = math.atan2(-rotation_trans_matrix[2, 0], math.sqrt(rotation_trans_matrix[2, 1]**2 + rotation_trans_matrix[2, 2]**2))
    # Yaw (rotation about z-axis)
    yaw = math.atan2(rotation_trans_matrix[1, 0], rotation_trans_matrix[0, 0])

    # Convert angles from radians to degrees
    roll = math.degrees(roll)
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)

    return (translation[0], translation[1], translation[2], roll, pitch, yaw)
    '''
    rv = np.genfromtxt(file_path)
    if np.any(np.isnan(rv)):
        raise OSError 
    #np.array_equal(array1, array2)
    '''
    file_path = os.path.join(self.results_dir,"rgb_images/transformations/experiment"+str(self.experiment)+ "_" + str(frame+1).zfill(4)+"_0.csv")
    rgb_trans = np.genfromtxt(file_path)
    if np.any(np.isnan(rv)): 
        raise OSError
    if not np.array_equal(rv,rgb_trans):
        print("rv: ",rv)
        print("rgb_trans:",rgb_trans)
        raise OSError
    '''
    return rv 

def get_frustrum_verts(robot_pose,max_range,fov,delta_theta=3): 
    min_d = 0.1; max_d = max_range*1.1 
    if len(robot_pose) == 6:
        min_theta = np.deg2rad((robot_pose[5]*180/np.pi) - (fov/2)) - np.deg2rad(delta_theta)
        max_theta = np.deg2rad((robot_pose[5]*180/np.pi) + (fov/2)) + np.deg2rad(delta_theta)
        if robot_pose[5] > 2*np.pi:
            print("robot_pose: ",robot_pose)
            raise OSError 
    elif len(robot_pose) == 3:
        '''
        if robot_pose[2] < 89:
            print("debugging the first timestep")
            raise OSError 
        '''
        min_theta = np.deg2rad(robot_pose[2]*180/np.pi - (fov/2)) - np.deg2rad(delta_theta) 
        max_theta = np.deg2rad(robot_pose[2]*180/np.pi + (fov/2)) + np.deg2rad(delta_theta)
        if robot_pose[2] > 2*np.pi:
            raise OSError
    x0 = robot_pose[0] + min_d*np.cos(min_theta)
    y0 = robot_pose[1] + min_d*np.sin(min_theta)
    x1 = robot_pose[0] + min_d*np.cos(max_theta)
    y1 = robot_pose[1] + min_d*np.sin(max_theta)
    x2 = robot_pose[0] + max_d*np.cos(min_theta)
    y2 = robot_pose[1] + max_d*np.sin(min_theta)
    x3 = robot_pose[0] + max_d*np.cos(max_theta)
    y3 = robot_pose[1] + max_d*np.sin(max_theta)
    return [(x0,y0),(x1,y1),(x2,y2),(x3,y3)]

def point_in_trapezoid(point, vertices):
    """
    Checks if a point falls within a trapezoid defined by its vertices.

    Args:
        point (tuple): (x, y) coordinates of the point to check.
        vertices (list of tuples): List of (x, y) coordinates of the trapezoid vertices.

    Returns:
        bool: True if the point is inside the trapezoid, False otherwise.
    """
    def is_left(p1, p2, p):
        x1,y1 = p1; x2,y2 = p2 #line segment

        x = p[0]; y = p[1] #point

        #return positive if p is to the left of the line segment 
        #return negative if p is to the right of the line segment 
        #if the result is 0 p is collinear with the line segment

        if abs(x2 - x1) > 0:
            m = (y2 - y1) / (x2 -x1) 
            b = y2 - m*x2 
            #find the point on the line segment that has the same y coordinate           
            x_lineSeg = (y-b)/m
            if x_lineSeg < x:
                #the point is right of the line segment
                return 1
            elif x_lineSeg == x:
                return 0 
            else:
                return -1 
        else:
            #print("this line is straight up and down")
            if x < x1:
                #print("point is left")
                return -1 
            else:
                #print("point is right")
                return 1 
            
    #print("this is point: ",point)
    wn = 0  # winding number
    n = len(vertices)
    #fig,ax = plt.subplots(1,4) 

    for i in range(n):
        #print("this is i:",i)
        p1 = vertices[i]
        #print("this is p1: ",p1)
        p2 = vertices[(i + 1) % n]
        #print("this is p2: ",p2)    
        if p1[1] <= point[1]:
            '''
            print("in the if...")
            print("this is_left(p1, p2, point): ",is_left(p1, p2, point))
            orientation = is_left(p1, p2, point)
            print("this is orientation: ",orientation)
            if  orientation < 0:
                bool = True 
            else: 
                bool = False
            ax[i].plot([p1[0],p2[0]],[p1[1],p2[1]],color="k")
            ax[i].scatter(point[0],point[1],color="r",marker="*")
            #self.ax.text(self.gt_trees[i,1],self.gt_trees[i,2],str(self.gt_trees[i,0]),fontsize=6,color="k",ha='center', va='center')
            ax[i].text(420, 240,str(orientation),color="k",ha="left",va = "center")
            if bool:
                ax[i].text(0.75, 0.95, f'Point is Left', ha='right', va='top', wrap=True, transform=ax[i].transAxes)
            else:
                ax[i].text(0.75, 0.95, f'Point is Right', ha='right', va='top', wrap=True, transform=ax[i].transAxes)

            '''
            
            if p2[1] > point[1] and is_left(p1, p2, point) < 0:
                #print("incrimenting the winding number")
                wn += 1
        else:
            '''
            ax[i].plot([p1[0],p2[0]],[p1[1],p2[1]],color="k")
            ax[i].scatter(point[0],point[1],color="r",marker="*")
            orientation = is_left(p1, p2, point) 
            print("this is orientation: ",orientation)
            if orientation < 0:
                bool = True 
            else:
                bool = False
            ax[i].text(420, 240,str(orientation),color="k",ha="left",va = "center")
            if bool:
                ax[i].text(0.75, 0.95, f'Point is Left', ha='right', va='top', wrap=True, transform=ax[i].transAxes)
            else:
                ax[i].text(0.75, 0.95, f'Point is Right', ha='right', va='top', wrap=True, transform=ax[i].transAxes)
                print("in the else...")
                print("this is_left(p1, p2, point):",is_left(p1, p2, point))
            '''
            
            if p2[1] <= point[1] and is_left(p1, p2, point) < 0:
                #print("decreasing the winding number")
                wn -= 1

    #plt.show(block=True)
    return wn != 0 


max_range = 250 

parameters = toml.load("/home/kristen/BetterFaster3.1/configs/carla.toml")
#exp,t,car_pose,carla_observations_t 
exp = 0 

orig_gt_car_traj = np.genfromtxt(os.path.join(parameters["results_dir"],"gt_car_poses/experiment"+str(exp + 1)+"_gt_car_pose.csv"),delimiter=",")
gt_car_traj = np.zeros((orig_gt_car_traj.shape[0],3))
gt_car_traj[:,0] = orig_gt_car_traj[:,0]
gt_car_traj[:,1] = orig_gt_car_traj[:,1]
gt_car_traj[:,2] = orig_gt_car_traj[:,5]

results_dir = "/media/kristen/easystore1/BetterFaster/kitti_carla_simulator/exp_results"  
obsd_clique_path = os.path.join(results_dir,"observation_pickles/experiment"+str(exp + 1)+"observed_cliques.pickle")

with open(obsd_clique_path,"rb") as handle:
    exp_observations = pickle.load(handle)
    print("exp_observations.keys(): ",exp_observations.keys())
    exp_observations = exp_observations[exp]

all_data_associations = {}
n_experiments = 11

for n in range(1,n_experiments+1):
    #experiment1data_association.csv
    data_associations_path = os.path.join(results_dir,"data_association/experiment"+str(n)+"data_association.csv")
    data_associations = np.genfromtxt(data_associations_path,delimiter=" ")
    all_data_associations[n] = data_associations

fig,ax = plt.subplots()
plt.ion() 

fov = 72 
img_width = 1392 
img_height = 1024 
sensor_noise_variance = 0.1 

for t in range(1000): 
    ax.clear() 
    carla_observations_t = exp_observations[t]  
    #print("carla_observations_t: ",carla_observations_t)
    car_pose = gt_car_traj[t,:] 
    gt_yaw = car_pose[2]
    sixd_cam_pose = np.zeros((6,))
    gt_range = 0.3; gt_yaw = -11.3645*(np.pi/180)
    sixd_cam_pose[0] = car_pose[0] + gt_range*np.cos(gt_yaw)
    sixd_cam_pose[1] = car_pose[1] + gt_range*np.sin(gt_yaw)
    sixd_cam_pose[2] =  1.7 
    sixd_cam_pose[-1] = np.deg2rad(car_pose[2]) 
    #robot_pose,max_range,fov 
    verts = get_frustrum_verts(sixd_cam_pose,max_range,fov)
    x0,y0 = verts[0]
    x1,y1 = verts[1]
    x2,y2 = verts[2]
    x3,y3 = verts[3]

    plt.plot([x1,x0],[y1,y0],'b')
    plt.plot([x0,x2],[y0,y2],'b')
    plt.plot([x2,x3],[y2,y3],'b')
    plt.plot([x3,x1],[y3,y1],'b')
    
    ax.scatter(car_pose[0],car_pose[1],color="k") 
    pointer_x = car_pose[0] + 2*np.cos(gt_yaw)  
    pointer_y = car_pose[1] + 2*np.sin(gt_yaw) 
    ax.plot([car_pose[0],pointer_x],[car_pose[1],pointer_y],"k") 

    print("there are {} lm ids".format(len(carla_observations_t.keys())))

    for lm_id in carla_observations_t.keys():
        world_pts_lm = [] 
        for feat_id in carla_observations_t[lm_id].keys():
            detx = {} 
            #self.all_data_associations = data_association 
            detx["clique_id"] = lm_id
            #detx["clique_id"] = get_reinitted_id(self.all_data_associations,self.experiment,lm_id) 
            #print("this is lm_id: ",lm_id)
            detx["feature_id"] = feat_id
            feat_loc = carla_observations_t[lm_id][feat_id]["feat_loc"] 
            feat_des = carla_observations_t[lm_id][feat_id]["feat_des"]
            detx["feature_des"] = feat_des
            #print("feat_loc: ",feat_loc)
            #exp,results_dir,frame,px_coord
            depth = get_depth(exp,results_dir,t,feat_loc)
            #print("depth:",depth)
            camera_pose_t = get_camera_pose(results_dir,exp,t) 
            #fov,width,height,camera_pose,depth,px 
            world_pt = get_world_pt(fov,img_width,img_height,camera_pose_t,depth,feat_loc)
            #print("world_pt: ",world_pt)
            
            world_pts_lm.append(world_pt) 
        
        world_pts_lm = np.array(world_pts_lm) 
        print("There are {} world pts".format(len(world_pts_lm)))
        X = world_pts_lm[:,0].reshape(-1,1) 
        y = world_pts_lm[:,1] 

        ransac = make_pipeline(PolynomialFeatures(degree=1), RANSACRegressor(residual_threshold=1.0,max_trials=1000,min_samples=2))  
        ransac.fit(X, y) 
        inlier_mask = ransac.named_steps['ransacregressor'].inlier_mask_
 
        inlier_pts = world_pts_lm[inlier_mask] 
        print("There are {} inlier pts".format(len(inlier_pts))) 

        for pt in inlier_pts: 
            bearing, range_ = get_range_bearing(car_pose,pt,sensor_noise_variance)
            ax.scatter(pt[0],pt[1],color="r",marker="*")  
            print("world_pt: ({},{})".format(pt[0],pt[1]))
            #sanity check 
            pt_x = car_pose[0] + range_*np.cos(bearing*(np.pi/180))
            pt_y = car_pose[1] + range_*np.sin(bearing*(np.pi/180)) 
            ax.scatter(pt_x,pt_y,color="g",marker="o") 
            print("sanity check: ({},{})".format(pt_x,pt_y)) 
            if range_ < max_range:
                print("bearing: {},range_: {}".format(bearing,range_))
                detx["bearing"] = bearing 
                detx["range"] = range_  
            delta_d = np.linalg.norm(pt[:2] - [pt_x,pt_y])
            if delta_d > 0.5:
                print("delta_d: ",delta_d) 
                raise OSError 
            
    plt.draw() 
    plt.pause(0.01)
    if len(carla_observations_t.keys()) > 0:
        input("Press Enter to Continue") 
        