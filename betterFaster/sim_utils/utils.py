import numpy as np 
import os 
import math 
import cv2 
import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation 
import re 

def get_observed_clique_ids(experiment_no,all_clique_feats,performance_tracker):
    #get_observed_clique_ids(exp,all_clique_feats,self.performance_tracker)
    all_observed_clique_ids = [x for x in all_clique_feats.keys()] 
    observed_clique_ids = []
    for id_ in all_observed_clique_ids: 
        #all_data_associations,n,id_
        if experiment_no > 1:
            orig_id = get_reinitted_id(performance_tracker.all_data_associations,experiment_no,id_)
        else:
            orig_id = id_
        if orig_id not in observed_clique_ids:
            #print("appending this id to observed clique ids:",orig_id)
            observed_clique_ids.append(orig_id)
    return observed_clique_ids 

def get_data_association(n_experiments,results_dir):
    data_association = {}
    for exp in range(n_experiments): 
        #exp0_data_association.csv
        filepath = "/data_associations/exp"+str(exp)+"_data_association.csv"
        arr = np.genfromtxt(results_dir + filepath,delimiter=" ")
        if np.isnan(arr).any():
            print("arr:",arr)
            raise OSError
        data_association[exp] = arr 
    return data_association

def get_reinitted_id(all_data_associations,n,id_,optional_exp=None): 
    print("getting reinitted_id!")

    if n == 1:
        print("n is one... returning this id: ",id_)
        if optional_exp is not None:
            print("in here!")
            lms_i = all_data_associations[n]
            matching_data_associations = all_data_associations[min(all_data_associations.keys())]
            if id_ in matching_data_associations[:,0]:
                idx = np.where(matching_data_associations[:,0] == id_)
                lm_id_pos = np.reshape(matching_data_associations[idx,1:],(2,1))
                if len(lm_id_pos) < 2:
                    raise OSError 
            else: 
                for k in all_data_associations.keys(): 
                    matching_data_associations = all_data_associations[k]
                    if id_ in matching_data_associations[:,0]: 
                        idx = np.where(matching_data_associations[:,0] == id_)
                        lm_id_pos = np.reshape(matching_data_associations[idx,1:],(2,1))
                        break 
            print("lms_i: ",lms_i)
            print("lm_id_pos: ",lm_id_pos)
            print("calling complicated function...")
            print("len(lm_id_pos):",len(lm_id_pos))
            row_idx = complicated_function(lms_i,lm_id_pos)
            #print("row_idx:",row_idx)
            if row_idx is None or lms_i[row_idx,0] == id_:
                print("made this condition...")
                lms_i = all_data_associations[n+1]
                if len(lm_id_pos) == 0:
                    raise OSError 
                row_idx = complicated_function(lms_i,lm_id_pos)
            reinitted_id = lms_i[row_idx,0]
            return reinitted_id
        
        else:
            return id_
    
    if all_data_associations is None: 
        print("all_data_associations is NONE")
        raise OSError 
        all_data_associations = {}
        for exp in range(n):
            data_association_filepath = "/home/kristen/BetterFaster3.0/sim_utils/data_association/exp"+str(exp)+"_data_association.csv"
            all_data_associations[exp+1] = np.genfromtxt(data_association_filepath)

    #print("all_data_associaitons: ",all_data_associations)

    #want to see if this landmark existed in experiments before this    
    if id_ in all_data_associations[min(all_data_associations.keys())][:,0]:
        if optional_exp is not None:
            print("in here!")
            matching_data_associations = all_data_associations[min(all_data_associations.keys())]
            idx = np.where(matching_data_associations[:,0] == id_)
            lm_id_pos = matching_data_associations[idx,1:]
            lms_i = all_data_associations[n]
            if len(lm_id_pos) == 0:
                raise OSError
            row_idx = complicated_function(lms_i,lm_id_pos)
            print("this is row_idx:",row_idx)
            if row_idx is None:
                lms_i = all_data_associations[n+1]
                print("this is lms_i:",lms_i)
                if len(lm_id_pos) == 0:
                    raise OSError
                row_idx = complicated_function(lms_i,lm_id_pos)
            reinitted_id = lms_i[row_idx,0]
            return reinitted_id
        else: 
            return id_
    
    if not id_ in all_data_associations[n][:,0]:
        if id_ in all_data_associations[n+1][:,0]:
            n += 1 

    idx = np.where(all_data_associations[n][:,0] == id_) 
    lm_id_pos = all_data_associations[n][idx,1:]
    reinitted_id = None 
    i = n - 1
    c = 0 
    while 1 <= i:
        #print("i:",i)
        lms_i = all_data_associations[i] 
        #print("lms_i ",lms_i)
        #print("lm_id_pos: ",lm_id_pos)
        if len(lm_id_pos) == 0:
            raise OSError
        row_idx = complicated_function(lms_i,lm_id_pos)
        if not row_idx is None:
            reinitted_id = lms_i[row_idx,0] 
        #print("reinitted_id: ",reinitted_id)
        '''
        else:
            if not reinitted_id is None: 
                #return reinitted_id
            reinitted_id = None 
        '''
        i -= 1
        '''
        print("row_idx: ",row_idx)
        if row_idx.size > 0:
            #this lm existed in the previous experiment 
            reinitted_id = lms_i[row_idx,0]
        else:
            #this lm did not exist in this experiment
            break
        except: 
            print("this landmark is not in the data associations....")
            return None 
        '''
    return int(reinitted_id )

def complicated_function(lms_i,lm_id_pos):
    print("lms_i: ",lms_i)
    print("lm_id_pos.shape: ",lm_id_pos.shape)
    if len(lm_id_pos.shape) > 2:
        print("lm_id_pos.shape: ",lm_id_pos.shape)
        lm_id_pos = lm_id_pos[0,0]
    print("lm_id_pos: ",lm_id_pos)
    i0 = [i for i,x in enumerate(lms_i[:,1]) if x == lm_id_pos[0]]
    i1 = [i for i,x in enumerate(lms_i[:,2]) if x == lm_id_pos[1]]
    idx = [x for x in i0 if x in i1] 
    if len(idx) > 1:
        raise OSError 
    elif len(idx) == 1: 
        return idx[0] 
    else:
        return None 

def extract_t_from_filename(filename): 
    match = re.search(r'_(\d+)_', filename)

    if match:
        extracted_integer = int(match.group(1))
        #print("Extracted integer:", extracted_integer)
    else:
        print("No match found.")
        raise OSError 
    
    return extracted_integer

def get_sim_length(results_dir,experiment_no): 
    dir_ = os.path.join(results_dir,"rgb_images")
    if not os.path.exists(dir_):
        dir_ = os.path.join(results_dir,"exp"+str(experiment_no)+"_results/rgb_images")
    if os.path.exists(dir_):
        exp_files = [x for x in os.listdir(dir_) if "experiment"+str(experiment_no) in x]
        return max([extract_t_from_filename(x) for x in exp_files]) + 1 
    else:
        return 500 
    

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


class simUtils(): 
    def __init__(self,exp,data_association,parameters,max_range=100):
        self.results_dir = parameters["results_dir"] 
        self.experiment = exp 
        self.width = parameters["vehicle_parameters"]["img_width"]
        self.height = parameters["vehicle_parameters"]["img_height"]
        self.fov = parameters["vehicle_parameters"]["fov"]
        self.max_range = max_range
        self.miss_detection_probability_function = parameters["betterTogether"]["miss_detection_probability_function"]
        self.sensor_noise_variance = parameters["betterTogether"]["sensor_noise_variance"]
        self.data_association = data_association 
        self.K = self.build_projection_matrix()

    def get_range_bearing(self,p0,p1,enable_noise_variance=True):
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
            noisy_landmark = p1 + np.random.normal(0,self.sensor_noise_variance,size=3)
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
            print("np.round(test_0,1):{}, np.round(x1,1):{}".format(np.round(test_0,1),np.round(x1,1)))
            delta_x = np.round(x1,1) - np.round(test_0,1) 
            print("delta_x: ",delta_x)
            print("test_0:{},test_1: {}".format(test_0,test_1))
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
            print("np.round(test_0,1):{}, np.round(x1,1):{}".format(np.round(test_0,1),np.round(x1,1)))
            delta_y = np.round(test_0,1) - np.round(x1,1)
            print("delta_y: ",delta_y)
            print("test_0:{},test_1: {}".format(test_0,test_1))
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

    def get_camera_pose(self,frame):
        #file_path = os.path.join(self.results_dir,"camera_transformations/experiment"+str(self.experiment)+"_frame"+str(frame).zfill(4)+".csv")
        file_path = os.path.join(self.results_dir,"depth_images/transformations/experiment"+str(self.experiment)+ "_" + str(frame+1).zfill(4)+"_20.csv")
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

    def build_projection_matrix(self):
        #if not isinstance(fov,int):
        if self.fov < 2*np.pi:
            #fov is probably in radians but we were expecting degrees
            self.fov = self.fov * (180/np.pi)
        focal = self.width / (2.0 * np.tan(self.fov * np.pi / 360.0))
        v_focal = self.height / (2.0 * np.tan(self.fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = focal 
        K[1, 1] = v_focal
        K[0, 2] = self.width / 2.0
        K[1, 2] = self.height / 2.0
        return K

    def get_depth(self,frame,px_coord,px_radius=3):
        depth_image = self.results_dir + "/depth_images/experiment" + str(self.experiment) +"_"+ str(frame).zfill(4) + "_20.png"
        #print("depth_image: ",depth_image)
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
    
    def get_world_pt(self,camera_pose,depth,px):
        #K = self.build_projection_matrix()
        cam_projector = CamProjector(self.K,camera_pose,depth)
        world_pt = cam_projector.project(px)
        return world_pt[:3]

    def convert_cam_pose(self,matrix): 
        """
        Extracts 6DoF pose [x, y, z, roll, pitch, yaw] from a 4x4 transformation matrix.

        Args:
            matrix (np.ndarray): 4x4 transformation matrix.

        Returns:
            list: 6DoF pose [x, y, z, roll, pitch, yaw].
        """
        # Extract translation [x, y, z]
        translation = matrix[:3, 3]

        # Extract rotation matrix
        rotation_matrix = matrix[:3, :3]

        # Extract roll, pitch, yaw from rotation matrix
        # Using conventions for XYZ Euler angles (roll-pitch-yaw)
        # Calculate pitch
        pitch = np.arcsin(-rotation_matrix[2, 0])

        # Check for edge cases
        if np.abs(rotation_matrix[2, 0]) != 1:
            # Calculate roll and yaw
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            # Gimbal lock: pitch is at ±90 degrees
            # Roll and yaw are not uniquely determined
            roll = 0  # Arbitrary value
            yaw = np.arctan2(-rotation_matrix[0, 1], rotation_matrix[1, 1])

        #print("this should be yaw in rads: ",yaw)
        # Convert angles to degrees
        roll = np.degrees(roll)
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        #print("this should be yaw in deg: ",yaw)

        return [translation[0], translation[1], translation[2], roll, pitch, yaw]

    def plot_frustrum(self,robot_pose): 
        min_d = 0.1; max_d = self.max_range
        min_theta = np.deg2rad(robot_pose[5] - (self.fov/2))
        max_theta = np.deg2rad(robot_pose[5] + (self.fov/2))
        if robot_pose[5] > 2*np.pi:
            raise OSError 
        x0 = robot_pose[0] + min_d*np.cos(min_theta)
        y0 = robot_pose[1] + min_d*np.sin(min_theta)
        x1 = robot_pose[0] + min_d*np.cos(max_theta)
        y1 = robot_pose[1] + min_d*np.sin(max_theta)
        x2 = robot_pose[0] + max_d*np.cos(min_theta)
        y2 = robot_pose[1] + max_d*np.sin(min_theta)
        x3 = robot_pose[0] + max_d*np.cos(max_theta)
        y3 = robot_pose[1] + max_d*np.sin(max_theta)
        #print("(x0,y0): {},(x1,y1): {}".format((x0,y0),(x1,y1)))
        #print("(x2,y2):{}, (x3,y3):{}".format((x2,y2),(x3,y3)))
        plt.plot([x0,x1],[y0,y1],'b')
        plt.plot([x1,x2],[y1,y2],'b')
        plt.plot([x2,x3],[y2,y3],'b')
        plt.plot([x3,x0],[y3,y0],'b')

    def get_frustrum_verts(self,robot_pose,delta_theta=3): 
        min_d = 0.1; max_d = self.max_range*1.1 
        if len(robot_pose) == 6:
            min_theta = np.deg2rad((robot_pose[5]*180/np.pi) - (self.fov/2)) - np.deg2rad(delta_theta)
            max_theta = np.deg2rad((robot_pose[5]*180/np.pi) + (self.fov/2)) + np.deg2rad(delta_theta)
            if robot_pose[5] > 2*np.pi:
                print("robot_pose: ",robot_pose)
                raise OSError 
        elif len(robot_pose) == 3:
            '''
            if robot_pose[2] < 89:
                print("debugging the first timestep")
                raise OSError 
            '''
            min_theta = np.deg2rad(robot_pose[2]*180/np.pi - (self.fov/2)) - np.deg2rad(delta_theta) 
            max_theta = np.deg2rad(robot_pose[2]*180/np.pi + (self.fov/2)) + np.deg2rad(delta_theta)
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
    
    def check_observation_w_gt(self,camera_pose_t,depth,feat_loc):
        gt_sixd_cam_pose  = self.convert_cam_pose(camera_pose_t)
        gt_sixd_cam_pose[5] = np.deg2rad(gt_sixd_cam_pose[5])
        print("gt_sixd_cam_pose:",gt_sixd_cam_pose)
        gt_world_pt = self.get_world_pt(camera_pose_t,depth,feat_loc)
        print("gt_world_pt:",gt_world_pt) 
        gt_verts = self.get_frustrum_verts(gt_sixd_cam_pose)
        if not point_in_trapezoid(gt_world_pt,gt_verts):
            print("this would not even be observable even if the pose estimate was better....")
            #this would not even be observable even if the pose estimate was better....
            #extract all observations is messed up 
            return False 
        else:
            print("the reason this is not observable is because the trajectory estiamte is off")
            #the reason this is not observable is because the trajectory estimate is off 
            #everything is ok
            return True 

    def reform_observations(self,t,car_pose,carla_observations_t,enable_miss_detection=True,enable_noise_variance=True): 
        #print("this is car pose: ",car_pose)
        #print("this is t: ",t)
        camera_pose_t = self.get_camera_pose(t)
        #print("camera_pose_t:",camera_pose_t)
        #so we cant use the gt location of the camera from the sim, we need to get the transfom using the rigid body transform 
        sixd_cam_pose = np.zeros((6,))
        gt_range = 0.3; gt_yaw = -11.3645*(np.pi/180)
        sixd_cam_pose[0] = car_pose[0] + gt_range*np.cos(gt_yaw)
        sixd_cam_pose[1] = car_pose[1] + gt_range*np.sin(gt_yaw)
        sixd_cam_pose[2] =  1.7 
        sixd_cam_pose[-1] = np.deg2rad(car_pose[2])

        #need to get camera pose, depth 
        observations = []

        #print("carla_observations_t.keys():",carla_observations_t.keys())
        for lm_id in carla_observations_t.keys():
            '''
            print("this is lm_id:",lm_id)
            print("carla_observations_t[lm_id]:",carla_observations_t[lm_id])
            print("these are the keys: ",carla_observations_t[lm_id].keys())
            '''
            for feat_id in carla_observations_t[lm_id].keys():
                detx = {}
                detx["clique_id"] = get_reinitted_id(self.data_association,self.experiment,lm_id) 
                print("this is lm_id: ",lm_id)
                detx["feature_id"] = feat_id
                #def get_world_pt(results_dir,experiment,frame,lm_id,width,height,fov,px_coord,camera_pose,depth):
                #print("carla_observations_t[lm_id][feat_id]:",carla_observations_t[lm_id][feat_id])
                #print("these are the keys:",carla_observations_t[lm_id][feat_id].keys())
                feat_loc = carla_observations_t[lm_id][feat_id]["feat_loc"] 
                feat_des = carla_observations_t[lm_id][feat_id]["feat_des"]
                detx["feature_id"]["feature_des"] = feat_des
                #print("feat_loc: ",feat_loc)
                depth = self.get_depth(t,feat_loc)
                #print("depth:",depth)
                world_pt = self.get_world_pt(camera_pose_t,depth,feat_loc)
                #print("world_pt: ",world_pt)
                verts = self.get_frustrum_verts(sixd_cam_pose)
                #print("verts: ",verts)
                if not point_in_trapezoid(world_pt,verts):
                    if np.linalg.norm(world_pt - sixd_cam_pose[:3]) < self.max_range*0.9:
                        #check if we would have been able to observe it if our trajectory was ground truth
                        if not self.check_observation_w_gt(camera_pose_t,depth,feat_loc):
                            print("this is not observable!")
                            idx = np.where(self.data_association[:,0] == lm_id)
                            if not lm_id in self.data_association[:,0]: 
                                continue 
                            print("self.data_association[lm_id,:]",self.data_association[idx,:])
                            print("world_pt: ",world_pt)
                            print("sixd_cam_pose: ",sixd_cam_pose)
                            print("feat_loc:",feat_loc)
                            print("depth:",depth)
                            #print("this is range: ",np.linalg.norm(world_pt - sixd_cam_pose[:3]))
                            #plot the car with points
                            fig,ax = plt.subplots()
                            ax.scatter(car_pose[0],car_pose[1],color="k")
                            if len(car_pose) == 6:
                                yaw = car_pose[5]
                                if np.abs(yaw) > 2*np.pi:
                                    yaw = np.deg2rad(yaw)
                            elif len(car_pose) == 3:
                                yaw = car_pose[2]
                                if np.abs(yaw) > 2*np.pi:
                                    yaw = np.deg2rad(yaw)
                            pointer_x = car_pose[0] + 2*np.cos(yaw)
                            pointer_y = car_pose[1] + 2*np.sin(yaw)
                            ax.plot([car_pose[0],pointer_x],[car_pose[1],pointer_y],"k")
                            #print("these are the verts: ",verts)
                            x0,y0 = verts[0]
                            x1,y1 = verts[1]
                            x2,y2 = verts[2]
                            x3,y3 = verts[3]
                            plt.plot([x1,x0],[y1,y0],'b')
                            plt.plot([x0,x2],[y0,y2],'b')
                            plt.plot([x2,x3],[y2,y3],'b')
                            plt.plot([x3,x1],[y3,y1],'b')
                            ax.scatter(world_pt[0],world_pt[1],color="r",marker="*")
                            #plt.show(block=True)
                            #raise OSError 
                            if not os.path.exists("observation_err_imgs"):
                                os.mkdir("observation_err_imgs") 
                            filename = os.path.join("observation_err_imgs","exp"+str(self.experiment)+"frame"+str(t).zfill(4)+".png")
                            print("writing {}".format(filename))
                            plt.savefig(filename)
                            plt.close()
                    else:
                        print("this vert is not observable because its {}m away but the max range is {}".format(np.linalg.norm(world_pt - sixd_cam_pose[:3]),self.max_range))
                        continue 
                
                #bearing, range_ = self.get_range_bearing(sixd_cam_pose,world_pt,car_pose,enable_noise_variance) #this bearing is in degrees
                bearing, range_ = self.get_range_bearing(car_pose,world_pt)
                '''
                if bearing is None or range_ is None:
                    fig,ax = plt.subplots()
                    ax.scatter(car_pose[0],car_pose[1],color="k")
                    pointer_x = car_pose[0] + 2*np.cos(np.deg2rad(car_pose[5]))
                    pointer_y = car_pose[1] + 2*np.sin(np.deg2rad(car_pose[5]))
                    ax.plot([car_pose[0],pointer_x],[car_pose[1],pointer_y],color="k")
                    ax.plot(sixd_cam_pose[0],sixd_cam_pose[1],color="b")
                    ax.scatter(world_pt[0],world_pt[1],color="r",marker="*")
                    self.plot_frustrum(car_pose)
                    ax.set_aspect('equal')
                    plt.show(block=True)
                    raise OSError
                '''
                if range_ < self.max_range:
                    #print("bearing: {},range_: {}".format(bearing,range_))
                    detx["bearing"] = bearing 
                    detx["range"] = range_
                    if enable_miss_detection:
                        if self.miss_detection_probability_function(range_) < np.random.random():
                            detection_var = 1 
                        else:
                            detection_var = 0
                    else:
                        detection_var = 1 
                    detx["detection"] = detection_var 
                '''
                else:
                    print("this is out of range!")
                '''
                observations.append(detx)

        return observations 