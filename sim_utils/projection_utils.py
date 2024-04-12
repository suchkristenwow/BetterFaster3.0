import numpy as np
import os 
import cv2 
#import open3d as o3d
from scipy.optimize import fsolve
#from geometry_utils import euclidean_distance
import matplotlib.pyplot as plt

def euclidean_distance(p0,p1):
    if len(p0) > 3:
        p0 = p0[:3]
    if len(p1) > 3:
        p1 = p1[:3]
    return np.linalg.norm(p1- p0)

def build_projection_matrix(w, h, fov):
    if not isinstance(fov,int):
        if fov < 2*np.pi:
            #fov is probably in radians but we were expecting degrees
            fov = fov * (180/np.pi)
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    v_focal = h / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = focal 
    K[1, 1] = v_focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        #point = np.array([loc.x, loc.y, loc.z, 1])
        point = np.array([loc[0],loc[1],loc[2],1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

def parse_carla_path(results_dir,experiments,sim_length):
    gt_traj = {}
    for run in range(experiments):
        traj_file = results_dir + "gt_car_poses/experiment" + str(run) + "_gt_car_pose.txt"
        #print(traj_file)
        exp_traj = []
        with open(traj_file) as f:
            lines = f.readlines()
            for line in lines:
                comma_idx = [i for i,x in enumerate(line) if x ==","]
                x_idx0 = line.index("x=")
                y_idx0 = line.index("y=")
                yaw_idx0 = line.index("yaw=")
                x = float(line[x_idx0+2:comma_idx[0]])
                y = float(line[y_idx0+2:comma_idx[1]])
                yaw = float(line[yaw_idx0+4:comma_idx[-1]])
                exp_traj.append([x,y,yaw])
        #downsample the traj measurements according to the sim length start:stop:step
        step = int(len(exp_traj) / sim_length)
        downsample_exp_traj = exp_traj[0:len(exp_traj):step]
        for i,t in enumerate(range(sim_length*run,sim_length*(run+1))):
            gt_traj[t] = downsample_exp_traj[i]
            gt_traj[t][2] = np.deg2rad(gt_traj[t][2])
    return gt_traj

def get_points_order(verts,pose):
    if min(verts[:,0]) < pose[0]:
        left_far = True 
    else:
        left_far = False 
    #x determines left/right
    #y determines front to back
    y0 = min(verts[:,1]); y1 = max(verts[:,1])
    if left_far:
        #max x is left
        if abs(y0 - pose[1]) < abs(y1 - pose[1]):
            #min y is front; max y is rear 
            fbl = [max(verts[:,0]),min(verts[:,1]),min(verts[:,2])]
            ftl = [max(verts[:,0]),min(verts[:,1]),max(verts[:,2])]
            fbr = [min(verts[:,0]),min(verts[:,1]),min(verts[:,2])]
            ftr = [min(verts[:,0]),min(verts[:,1]),max(verts[:,2])]
            rbl = [max(verts[:,0]),max(verts[:,1]),min(verts[:,2])]
            rbr = [min(verts[:,0]),max(verts[:,1]),min(verts[:,2])]
            rtl = [max(verts[:,0]),max(verts[:,1]),max(verts[:,2])]
            rtr = [min(verts[:,0]),max(verts[:,1]),max(verts[:,2])]
        else:
            #min y is rear, max y is front 
            fbl = [max(verts[:,0]),max(verts[:,1]),min(verts[:,2])]
            fbr = [min(verts[:,0]),max(verts[:,1]),min(verts[:,2])]
            rbl = [max(verts[:,0]),min(verts[:,1]),min(verts[:,2])]
            rbr = [min(verts[:,0]),min(verts[:,1]),min(verts[:,2])]
            ftl = [max(verts[:,0]),max(verts[:,1]),max(verts[:,2])]
            ftr = [min(verts[:,0]),max(verts[:,1]),max(verts[:,2])]
            rtl = [max(verts[:,0]),min(verts[:,1]),max(verts[:,2])]
            rtr = [min(verts[:,0]),min(verts[:,1]),max(verts[:,2])]
    else:
        #max x is right
        if abs(y0 - pose[1]) < abs(y1 - pose[1]):
            #min y is front; max y is rear 
            fbl = [min(verts[:,0]),min(verts[:,1]),min(verts[:,2])]
            fbr = [max(verts[:,0]),min(verts[:,1]),min(verts[:,2])]
            rbl = [min(verts[:,0]),max(verts[:,1]),min(verts[:,2])]
            rbr = [max(verts[:,0]),max(verts[:,1]),min(verts[:,2])]
            ftl = [min(verts[:,0]),min(verts[:,1]),max(verts[:,2])]
            ftr = [max(verts[:,0]),min(verts[:,1]),max(verts[:,2])]
            rtl = [min(verts[:,0]),max(verts[:,1]),max(verts[:,2])]
            rtr = [max(verts[:,0]),max(verts[:,1]),max(verts[:,2])]
        else:
            #max y is front; min y is rear
            fbl = [min(verts[:,0]),max(verts[:,1]),min(verts[:,2])]
            fbr = [max(verts[:,0]),max(verts[:,1]),min(verts[:,2])]
            rbl = [min(verts[:,0]),min(verts[:,1]),min(verts[:,2])]
            rbr = [max(verts[:,0]),min(verts[:,1]),min(verts[:,2])]
            ftl = [min(verts[:,0]),max(verts[:,1]),max(verts[:,2])]
            ftr = [max(verts[:,0]),max(verts[:,1]),max(verts[:,2])]
            rtl = [min(verts[:,0]),min(verts[:,1]),max(verts[:,2])]
            rtr = [max(verts[:,0]),min(verts[:,1]),max(verts[:,2])]
    return fbl,fbr,ftr,ftl,rbl,rbr,rtr,rtl

def find_boundingbox_intersection(bounding_box_verts,p0,p1,diag,experiment,frame,lm_id,depth):
    min_x = min(bounding_box_verts[:,0])
    max_x = max(bounding_box_verts[:,0])
    min_y = min(bounding_box_verts[:,1])
    max_y = max(bounding_box_verts[:,1])
    min_z = min(bounding_box_verts[:,2])
    max_z = max(bounding_box_verts[:,2])
    centroid = np.mean(bounding_box_verts,0)

    fbl,fbr,ftr,ftl,rbl,rbr,rtr,rtl = get_points_order(bounding_box_verts,p0)

    if not os.path.exists("debug"):
        os.mkdir("debug")

    with open("debug/bb_corners_experiment"+str(experiment)+"frame"+str(frame)+"lm"+str(lm_id)+".txt","w") as f:
        f.write(str(p0))
        f.write(str([fbl,fbr,ftr,ftl,rbl,rbr,rtr,rtl]))
        f.close()

    p0 = p0[:3]
    #find the closest intersection with one of the six planes defining the box 
    valid_intersections = []
    all_intersections = []
    for i in range(6):
        if i == 0:
            p_n0 = [1,0,0]
            p_c0 = np.mean([ftr,rbr],0)
        if i == 1:
            p_n0 = [0,1,0]
            p_c0 = np.mean([ftl,fbr],0)
        if i == 2:
            p_n0 = [0,0,1]
            p_c0 = np.mean([ftl,rtr],0)
        if i == 3:
            p_n0 = [-1,0,0]
            p_c0 = np.mean([ftl,rbl],0)
        if i == 4:
            p_n0 = [0,-1,0]
            p_c0 = np.mean([rtr,rbr],0)
        if i == 5:
            p_n0 = [0,0,-1]
            p_c0 = np.mean([fbl,rbr],0)
        u = p1 - p0
        d = np.dot(p_n0,u)
        if abs(d) > 1e-6:
            w = p0 - p_c0
            fac = -np.dot(p_n0,w) / d
            u = u * fac
            u += p0
            #print("u: ",u)
            all_intersections.append(u)
            if euclidean_distance(p0,u) == depth:
                return u 
            if min_x - diag*.1 <= u[0] <= max_x + diag*.1:
                if min_y - diag*.1 <= u[1] <= max_y + diag*.1:
                    if min_z - diag*.1 <= u[2] <= max_z + diag*.1:
                        valid_intersections.append(u)
                        continue
            if euclidean_distance(u,centroid) < diag:
                valid_intersections.append(u)

    #get minimum distance between p0 and the intersection
    if len(valid_intersections) > 1:
        d = [np.sqrt((x[0] - p0[0])**2 + (x[1] - p0[1])**2 + (x[2] - p0[2])**2) for x in valid_intersections] 
        i = np.argmin(d)
        return valid_intersections[i]
    else:
        if len(valid_intersections) > 0:
            return valid_intersections[0]   
        else:
            d = [np.sqrt((x[0] - p0[0])**2 + (x[1] - p0[1])**2 + (x[2] - p0[2])**2) for x in all_intersections] 
            i = np.argmin(d)
            return all_intersections[i]

def get_depth(image,px_coord):
    '''
    depth_image = results_dir + "depth_images/experiment" + str(experiment) +"_"+ str(frame).zfill(4) + "_20.png"
    #print("depth_image: ",depth_image)
    im = cv2.imread(depth_image)
    '''
    #print(im.shape)
    x = int(px_coord[0]); y = int(px_coord[1])
    C = image[y][x]
    B = C[0]; G = C[1]; R = C[2]
    normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)

    return normalized * 1000

def get_camera_pose(car_pose,camera_offset):    
    tmp = car_pose #car pose is x,y,bearing
    camera_pose = tmp 
    camera_pose[0] += camera_offset[0]*np.cos(car_pose[2]*(np.pi/180)) + camera_offset[1]*np.sin(car_pose[2]*(np.pi/180))
    camera_pose[1] += camera_offset[0]*np.sin(car_pose[2]*(np.pi/180)) + camera_offset[1]*np.cos(car_pose[2]*(np.pi/180))
    camera_pose[2] = camera_offset[2]
    camera_pose = np.array(camera_pose)
    return camera_pose

def normalize(v):
    return v / np.linalg.norm(v)

def get_projection_params(results_dir,experiment):
    with open(results_dir + "lidar2cam_transformations/" + str(experiment) + "lidar_to_cam0.txt") as f:
        lines = f.readlines()
        data = [float(x) for x in lines[1].split(" ")]
        data = np.array(data)
        data = np.reshape(data,(3,4))
        R = data[:,:3]
        t = data[:,3]
        return R,t 

def get_world_pt(results_dir,experiment,frame,lm_id,width,height,fov,px_coord,camera_pose,depth):
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio) # left, top, right, bottom
    K = build_projection_matrix(width, height, fov)
    obj_coord_files = [x for x in os.listdir(results_dir + "/lm_pts/") if "experiment"+str(experiment) in x and "frame"+str(frame).zfill(4) in x and str(lm_id) in x]
    if len(obj_coord_files) > 0:
        if len(obj_coord_files) > 1:
            raise OSError("What")
        obj_coords = np.genfromtxt(results_dir + "/lm_pts/" + obj_coord_files[0],delimiter=",")

    intrinsics_result = get_world_point_intrinsics(px_coord,K,camera_pose)
    
    #print("result from get_world_point_intrinsics: ",result)

    diag = np.sqrt((min(obj_coords[:,0]) - max(obj_coords[:,0]))**2 + (min(obj_coords[:,1]) - max(obj_coords[:,1]))**2 + (min(obj_coords[:,2]) - max(obj_coords[:,2]))**2)
        
    '''
    print("min_x: {} max_x: {}".format(min(obj_coords[:,0]),max(obj_coords[:,0])))
    print("min_y: {} max_y: {}".format(min(obj_coords[:,1]),max(obj_coords[:,1])))
    print("min_z: {} max_z: {}".format(min(obj_coords[:,2]),max(obj_coords[:,2])))
    '''

    if not check_bb(intrinsics_result,obj_coords,diag):
        centroid = np.mean(obj_coords,axis=0)

        y = np.linspace(screen[1], screen[3], height)
        x = np.linspace(screen[0], screen[2], width)

        i = int(px_coord[0])    
        j = int(px_coord[1])

        pixel = np.array([x[i], y[j], 0])

        direction = normalize(pixel - camera_pose[:3])
        
        result = find_boundingbox_intersection(obj_coords, camera_pose, direction, diag,experiment, frame, lm_id,depth)
        
        if not euclidean_distance(result,camera_pose) == depth:
            result = find_matching_depth_pt(camera_pose,result,diag,depth,obj_coords)
        #print("this is result: ",result)
        return result 

    else:
        #print("intrinsics_result worked!")
        if not euclidean_distance(intrinsics_result,camera_pose) == depth:
            intrinsics_result = find_matching_depth_pt(camera_pose,intrinsics_result,diag,depth,obj_coords)
        #print("this is intrinsics_result: ",intrinsics_result)
        return intrinsics_result

def find_matching_depth_pt(p0,p1,r,depth,obj_coords):
    pts = [] 
    radius = np.random.uniform(.1,r,1000)
    #print("this is depth: ",depth)
    for i in range(1000):
        a=np.random.randn(3)
        pts.append((a/sum(a*a)**.5)*radius[i] + p1)
    pts = np.array(pts)
    dists = [euclidean_distance(x,p0) for x in pts]
    idx = [i for i,x in enumerate(dists) if depth - 0.1 < x < depth + 0.1]
    if len(idx) == 0:
        #print("No valid projection")
        return []
    else:
        if len(idx) == 1:
            i0 = idx[0]
            #print("found result: ",pts[i0,:])
            return pts[i0,:]
        elif len(idx) > 1:
            i0 = np.argmin(dists)
            #print("found result: ",pts[i0,:])
            return pts[i0,:]


def get_world_point_intrinsics(px_coord,K,pose):
    #[R | t] is the camera pose wrt the world frame 
    #print("this is pose: ",pose)
    angle = pose[2]
    r = [[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]
    #print("this is r:",r)
    R = np.array(r)
    t = [pose[0],pose[1],1.7] 
    p = np.array([px_coord[0],px_coord[1],1]).T
    pc = np.linalg.inv(K) @ p 
    pw = t + (R@pc)
    cam = np.array([0,0,0]).T 
    cam_world = t + R @ cam 
    #find a ray from camera to 3d point 
    v = pw - cam_world 
    unit_vector = v / np.linalg.norm(v)

    p3D = cam_world + unit_vector 
    return p3D

def check_bb(u,obj_coords,diag):
    min_x = min(obj_coords[:,0]); max_x = max(obj_coords[:,0])
    min_y = min(obj_coords[:,1]); max_y = max(obj_coords[:,1])
    min_z = min(obj_coords[:,2]); max_z = max(obj_coords[:,2])

    if min_x - diag*.1 <= u[0] <= max_x + diag*.1:
        if min_y - diag*.1 <= u[1] <= max_y + diag*.1:
            if min_z - diag*.1 <= u[2] <= max_z + diag*.1:
                return True 

    return False 