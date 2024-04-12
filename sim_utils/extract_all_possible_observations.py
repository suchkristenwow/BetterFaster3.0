import os 
import argparse
import time 
import numpy as np
import cv2
from scipy import stats as st
import pickle 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.linear_model import RANSACRegressor
from scipy.spatial import KDTree
import statistics
from PIL import Image 
from sklearn.cluster import DBSCAN
import matplotlib.patches as patches
from datetime import datetime, date
from extract_gt_car_traj import get_gt_car_data


def downsample_image(img_path,output_filename,desired_width=800,desired_height=600): 
    image = Image.open(img_path)
    # Calculate the aspect ratio of the original image
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    # Calculate the new dimensions while maintaining the aspect ratio
    if original_width > original_height:
        new_width = desired_width
        new_height = int(desired_width / aspect_ratio)
    else:
        new_height = desired_height
        new_width = int(desired_height * aspect_ratio)

    # Resize the image to fit the desired number of pixels
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Save or display the resized image
    resized_image.save(output_filename)
    return resized_image 

def closest_points_kdtree(points, reference_point, k=10):
    kdtree = KDTree(points)
    closest_indices = kdtree.query(reference_point, k=k)[1]
    closest = [points[i] for i in closest_indices]
    return closest

def get_n_clusters(patch_coords,  max_clusters=5, num_simulations=10):
    wcss_values = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(patch_coords)
        wcss_values.append(kmeans.inertia_)

    gap_values = []
    for k in range(1, max_clusters + 1):
        gap = wcss_values[0] - wcss_values[k - 1]
        reference_gaps = []
        for _ in range(num_simulations):
            # Generate random data with the same shape as patch_coords
            random_data = np.random.rand(*patch_coords.shape)
            random_kmeans = KMeans(n_clusters=k, random_state=0)
            random_kmeans.fit(random_data)
            random_wcss = random_kmeans.inertia_
            reference_gaps.append(random_wcss)
        avg_reference_gap = np.mean(reference_gaps)
        gap_values.append(np.log(avg_reference_gap) - np.log(gap))

    return np.argmax(gap_values) + 1

def continuity_check(pixel,coordinates_set,delta_x=0): 
    coordinates_set = [(int(x[0]),int(x[1])) for x in coordinates_set]
    x, y = pixel
    for dx in [-1 - delta_x, 0, 1 + delta_x]:
        for dy in [-1 - delta_x, 0, 1 + delta_x]:
            if (int(x + dx), int(y + dy)) in coordinates_set:
                return True
    return False

def cluster_reject(patch_coords,lm_type):
    #trying to cluster the points such that we can guess which object is actually the landmark 
    #presumably, the largest portion of the bounding box is the landmark 
    #n_clusters = get_n_clusters(np.array(patch_coords))
    #print("n_clusters:",n_clusters)

    kmeans = KMeans(n_clusters=2, random_state=0)
    cluster_labels = kmeans.fit_predict(patch_coords)

    # Step 2: Identify the largest cluster
    largest_cluster = np.argmax(np.bincount(cluster_labels))

    # Step 3: Use RANSAC or another outlier removal method on the largest cluster
    largest_cluster_indices = np.where(cluster_labels == largest_cluster)[0]

    #print("{} points out of {} fit in the largest cluster".format(len(largest_cluster_indices),len(patch_coords)))

    patch_coords = np.array(patch_coords)
    coordinates = patch_coords[largest_cluster_indices]
    #print("coordinates.shape".format(coordinates.shape))

    # Define the DBSCAN clustering model
    dbscan = DBSCAN(eps=1, min_samples=2)  # Adjust eps and min_samples as needed

    # Fit the model to your data
    labels = dbscan.fit_predict(coordinates)

    # Separate main cluster (label -1 represents outliers)
    filtered_features= coordinates[labels != -1]
    #print("there are {} outliers".format(np.count_nonzero(labels == -1)))
    print("there are {} features after outlier removal".format(filtered_features.shape[0]))

    if len(filtered_features) == 0:
        raise OSError 

    return filtered_features

def depth_filtering(patch_coords,depth_img): 
    depth_vals = []
    for coord in patch_coords:
        i = int(coord[0]); j = int(coord[1])
        R = depth_img[j,i][0]; G = depth_img[j,i][1]; B = depth_img[j,i][2]
        depth_val = R + G*256 + B*256*256
        norm_depth_val = depth_val / (256*256*256 -1)
        depth_vals.append(norm_depth_val)

    depth_vals = np.array(depth_vals)
    depth_vals = depth_vals.reshape(-1,1)

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(depth_vals)

    # Get the cluster centers and labels
    #cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    # Step 2: Identify the largest cluster
    largest_cluster = np.argmax(np.bincount(cluster_labels))

    # Step 3: Use RANSAC or another outlier removal method on the largest cluster
    largest_cluster_indices = np.where(cluster_labels == largest_cluster)[0]

    filtered_features = patch_coords[largest_cluster_indices]

    return filtered_features 

def get_associated_features(img_path,orb_feat_coords,is_img,lm_id,lm_type,bb_coords,filename,gt_car_traj,data_associations):
    '''
    Finds the ORB features that are associated to a landmark. First, we take all the orb features in the landmark bounding box 
    Then we filter by the semantic label 
    '''
    patch_coords = []

    xmin = bb_coords[0]; ymin = bb_coords[1]
    xmax = bb_coords[2]; ymax = bb_coords[3]

    too_low = []
    tmp = []
    isVals = []
    wrong_type = []
    for orb_feat_coord in orb_feat_coords:
        if xmin <= orb_feat_coord[0] <= xmax:
            if ymin <= orb_feat_coord[1] <= ymax:
                #bb_points += 1 
                is_val = is_img[int(orb_feat_coord[1]),int(orb_feat_coord[0])]
                #print("is_val: ",is_val)
                tmp.append(orb_feat_coord)
                if is_val[0] == lm_type:
                    if lm_type == 9: #trees
                        #print("found a tree!")
                        if is_val[1] in range (175,186):
                            if is_val[2] in range(155,160):
                                #print("this color is wrong")
                                continue 
                        if is_val[1] in range(210,230):
                            if is_val[2] in range(175,185):
                                #print("this color is wrong")
                                continue 
                        if is_val[1] in range(120,125): 
                            if is_val[2] in range(85,95):
                                #print("this color is wrong")
                                continue 
                        #tree points should be up high 
                        bb_height = ymax - ymin
                        
                        if ymin + bb_height*.5 < orb_feat_coord[1]:
                            #(557,383)
                            '''
                            print("ymin + bb_height*.5: ",ymin + bb_height*.4)
                            print("orb_feat_coord: ",orb_feat_coord)
                            print("this is too low")
                            '''
                            if ymin == 0: 
                                #the landmark is cut off at the top
                                bb_width = xmax - xmin 
                                if orb_feat_coord[0] <= xmin+bb_width*0.25 or xmax - bb_width*0.25 <= orb_feat_coord[0]:
                                    #this landmark is on the fringes of the image which isnt right ....
                                    continue 
                            too_low.append(orb_feat_coord)
                            continue 

                    if lm_type == 20: #cones 
                        if orb_feat_coord[1] < is_img.shape[0]/2:
                            print("this is probably too high up to be a cone")
                            print("filename: ",filename)
                            raise OSError 
                        #print("found a cone!")
                        if not 203 <= is_val[2] <= 205:
                            #print("is_val: ",is_val)
                            #print("this color is bad")
                            continue 
                    
                    #print("appending patch coords: ",orb_feat_coord)
                    patch_coords.append(orb_feat_coord)
                    isVals.append(is_val)

                else: 
                    wrong_type.append(orb_feat_coord)

    if len(patch_coords) == 0:
        #print("patch_coords are zero!")
        bb_height = ymax - ymin 
        bb_width = xmax - xmin 
        bb_area = bb_height * bb_width 
        if bb_area > 7e3:
            idx = np.where(data_associations[:,0] == lm_id)
            gt_lm_pos = data_associations[idx,1:] 
            d = np.linalg.norm(gt_car_traj[:2] - gt_lm_pos[:2])
            if d < 100:
                '''
                print("gt_lm_pos: ",gt_lm_pos)
                print("gt_car_traj: ",gt_car_traj)
                print("this is d: ",d)
                print("this is sus")
                print("this is bb_area but there are no features associated to this lm: ",bb_area) 
                '''
                fig, ax = plt.subplots()
                #Debug feature detection images
                img = cv2.imread(img_path)
                #plt.imshow(img)
                plt.imshow(is_img)
                rectangle = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rectangle)
                orb_feat_coords = np.array(orb_feat_coords)
                plt.scatter(orb_feat_coords[:,0],orb_feat_coords[:,1],color="blue",s=2) #all orb feat coords 
                tmp = np.array(tmp)
                if tmp.shape[0] > 1:
                    plt.scatter(tmp[:,0],tmp[:,1],color="red",s=2) #orb coords in the bounding box 
                else:
                    if len(tmp) > 0:  
                        if len(tmp) == 1:
                            tmp = tmp[0]
                        print("tmp:",tmp)
                        plt.scatter(tmp[0],tmp[1],color="red",s=2)
                too_low = np.array(too_low) 
                wrong_type = np.array(wrong_type)
                if too_low.shape[0] > 1 and wrong_type.shape[0] > 1:
                    plt.scatter(too_low[:,0],too_low[:,1],color="k",marker="*") #too low 
                    plt.scatter(wrong_type[:,0],wrong_type[:,1],color="orange",marker="^") #wrong type 
                    #plt.show(block=True)
                    plt.axis("off")
                    if not os.path.exists("empty_plot_coords"):
                        os.mkdir("empty_plot_coords") 
                    print("writing: {}".format("empty_plot_coords/"+filename+"png"))
                    plt.savefig("empty_plot_coords/"+filename+"png") 
                    plt.close() 
        '''
            else:
                print("this is hella far... its probably just obscured")
        else:
            print("bb_area: {}. this bb is very smol".format(bb_area))
        '''
        return [] 

    orig_patch_coords = np.array(patch_coords)

    if np.shape(orig_patch_coords)[0] < 2: 
        print("orig_path_coords: ",orig_patch_coords)
        print("this is orig_patch_coords: ",orig_patch_coords.shape)
        return []
    
    patch_coords = np.array(patch_coords)
    orb_feat_coords = np.array(orb_feat_coords)

    fig, ax = plt.subplots()
    #Debug feature detection images
    img = cv2.imread(img_path)
    #plt.imshow(img)
    plt.imshow(is_img)
    #corners of the bounding box of the landmark
    rectangle = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rectangle)

    #scatter the orb features 
    orb_feat_coords = np.array(orb_feat_coords)
    plt.scatter(orb_feat_coords[:,0],orb_feat_coords[:,1],color="k",s=2) #black
    #orb features in the bounding box
    plt.scatter(orig_patch_coords[:,0],orig_patch_coords[:,1],color="b",s=2) #blue 

    #patch_coords = cluster_reject(patch_coords,lm_type)
    #print("this is the len of patch coords: ",len(patch_coords))

    #scatter the associated features 
    plt.scatter(patch_coords[:,0],patch_coords[:,1],color="magenta",s=3)

    if not os.path.exists("debugFeatureDetection_images"): 
        os.mkdir("debugFeatureDetection_images")

    print("writing ","debugFeatureDetection_images/"+filename+"jpg")
    plt.axis("off")
    plt.savefig("debugFeatureDetection_images/"+filename+"jpg")
    plt.close() 

    return patch_coords

def extract_orb_feats(results_dir,filename,output_filename): 
    #print("extracting orb features of {}".format(filename))
    image = cv2.imread(filename,flags=0)
    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=2500)
    
    # Find the keypoints and descriptors with ORB
    kp, des = orb.detectAndCompute(image, None)
    kp_arr = []
    for k in kp:
        kp_arr.append([k.pt[0],k.pt[1]])
    kp_arr = np.array(kp_arr)
    if not os.path.exists(os.path.join(results_dir,"orb_feats")): 
        os.mkdir(os.path.join(results_dir,"orb_feats"))
    
    if not os.path.exists(os.path.join(results_dir,"orb_feats/orb_kps")): 
        os.mkdir(os.path.join(results_dir,"orb_feats/orb_kps"))
    if not os.path.exists(os.path.join(results_dir,"orb_feats/orb_des")): 
        os.mkdir(os.path.join(results_dir,"orb_feats/orb_des"))

    file = os.path.splitext(os.path.basename(output_filename))[0]

    output_name = os.path.join(results_dir,"orb_feats/orb_kps" + file + ".csv")
    if "orb_kps" not in output_name:
        raise OSError 

    #print("writing {}".format(os.path.join(results_dir,"orb_feats/orb_kps",file + ".csv")))

    np.savetxt(os.path.join(results_dir,"orb_feats/orb_kps",file + ".csv"),kp_arr)
    np.savetxt(os.path.join(results_dir,"orb_feats/orb_des",file + ".csv"),des)

def parse_ids(results_dir,exp,lm_type): 
    filename = os.path.join(results_dir,lm_type+"_ids/experiment"+str(exp)+lm_type+"_ids_.txt")
    ids = []
    with open(filename,"r") as handle: 
        for line in handle.readlines():
            ids.append(int(line))
    return ids 

def get_rgb_img_path(dir_,x):
    experiment_idx = x.index("experiment")
    frame_idx = x.index("frame")
    exp = x[experiment_idx+10:frame_idx]
    #print("exp: ",exp)
    if "tree" in x:
        lm_idx = x.index("tree")
    else: 
        lm_idx = x.index("cone")
    frame_no = x[frame_idx+5:lm_idx] 
    #print("frame_no :",frame_no)
    rgb_img = os.path.join(dir_,"rgb_images/experiment"+exp+"_"+frame_no+"_0.png")
    if not os.path.exists(rgb_img):
        #print("rgb_img: ",rgb_img)
        return None 

    return rgb_img 

def save_boundingbox_image(dir_,rgb_img,x): 
    print("rgb_img:",rgb_img)
    if not os.path.exists(rgb_img):
        raise OSError
    img = cv2.imread(rgb_img); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #print("read the image successfuly")
    plt.imshow(img)
    #print("os.path.join(dir_,lm_2dpts/+ x): ",os.path.join(dir_,"lm_2dpts/" + x))
    bounding_box_coords = np.genfromtxt(os.path.join(dir_,"lm_2dpts/" + x))
    #print("bounding_box_coords: ",bounding_box_coords)
    #x_min,y_min,x_max,y_max
    xmin = bounding_box_coords[0]; ymin = bounding_box_coords[1]
    xmax = bounding_box_coords[2]; ymax = bounding_box_coords[3]
    plt.plot([xmax,xmax],[ymax,ymin],color="r") #bottom_right -> top right
    plt.plot([xmax,xmin],[ymin,ymin],color="r") #top_right -> top left
    plt.plot([xmin,xmin],[ymin,ymax],color="r") #top left -> bottom left
    plt.plot([xmin,xmax],[ymax,ymax],color="r") #bottom left -> bottom right 
    plt.axis("off")
    plt.savefig(os.path.join(dir_,"lm_2dpts/" + x[:-4]+".png"))
    plt.close()
    #plt.show()
    print("writing: ",os.path.join(dir_,"lm_2dpts/" + x[:-4]+".png"))

def get_most_recent(dir_,exp):
    '''
    with open(os.path.join("intermediate_pickles",str(t) + "experiment"+ str(exp +1)+ "observed_cliques.pickle"),"wb") as handle:
        pickle.dump(obsd_cliques,handle,protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join("intermediate_pickles",str(t) + "experiment"+ str(exp +1)+ "all_clique_feats.pickle"),"wb") as handle:
        pickle.dump(all_clique_feats,handle,protocol=pickle.HIGHEST_PROTOCOL)
    '''
    files = [x for x in os.listdir(dir_) if "experiment"+str(exp+1) in x and "observed_cliques.pickle" in x]
    file_modification_times = [(file, os.path.getmtime(os.path.join(dir_, file))) for file in files]
    # Sort the files by modification time in descending order
    file_modification_times.sort(key=lambda x: x[1], reverse=True)
    if len(file_modification_times) > 1:
        # The second most recently edited file is at index 1 (index 0 is the most recently edited file)
        second_most_recent_file = file_modification_times[0][0]
        
        # Get today's date
        today = date.today()

        # Convert modification time of the second most recent file to datetime object
        modification_time = datetime.fromtimestamp(file_modification_times[1][1]).date()

        if modification_time == today: 
            print("modification time is today... loading in observations")
            #filename = os.path.basename(second_most_recent_file)
            idx = second_most_recent_file.index("experiment")
            second_most_recent_file_t = int(second_most_recent_file[:idx])  
            with open(os.path.join(dir_,second_most_recent_file),"rb") as handle:
                obsd_cliques = pickle.load(handle) 
            if not 0 in obsd_cliques[exp].keys():
                raise OSError 
            clique_feats_file = os.path.join(dir_,str(second_most_recent_file_t)+"experiment"+ str(exp +1)+ "all_clique_feats.pickle")
            if not os.path.exists(clique_feats_file):
                raise OSError
            with open(clique_feats_file,"rb") as handle:
                all_clique_feats = pickle.load(handle)
            last_processed_t = second_most_recent_file_t
        else:
            obsd_cliques = {} 
            all_clique_feats = {} 
            last_processed_t = 0 
    else:
        obsd_cliques = {} 
        all_clique_feats = {} 
        last_processed_t = 0 

    return obsd_cliques, all_clique_feats, last_processed_t

def get_second_most_recent(dir_,exp):
    '''
    with open(os.path.join("intermediate_pickles",str(t) + "experiment"+ str(exp +1)+ "observed_cliques.pickle"),"wb") as handle:
        pickle.dump(obsd_cliques,handle,protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join("intermediate_pickles",str(t) + "experiment"+ str(exp +1)+ "all_clique_feats.pickle"),"wb") as handle:
        pickle.dump(all_clique_feats,handle,protocol=pickle.HIGHEST_PROTOCOL)
    '''
    files = [x for x in os.listdir(dir_) if "experiment"+str(exp+1) in x and "observed_cliques.pickle" in x]
    file_modification_times = [(file, os.path.getmtime(os.path.join(dir_, file))) for file in files]
    # Sort the files by modification time in descending order
    file_modification_times.sort(key=lambda x: x[1], reverse=True)
    if len(file_modification_times) > 1:
        # The second most recently edited file is at index 1 (index 0 is the most recently edited file)
        second_most_recent_file = file_modification_times[1][0]
        
        # Get today's date
        today = date.today()

        # Convert modification time of the second most recent file to datetime object
        modification_time = datetime.fromtimestamp(file_modification_times[1][1]).date()

        if modification_time == today: 
            print("modification time is today... loading in observations")
            #filename = os.path.basename(second_most_recent_file)
            idx = second_most_recent_file.index("experiment")
            second_most_recent_file_t = int(second_most_recent_file[:idx])  
            with open(os.path.join(dir_,second_most_recent_file),"rb") as handle:
                obsd_cliques = pickle.load(handle) 
            if not 0 in obsd_cliques[exp].keys():
                raise OSError 
            clique_feats_file = os.path.join(dir_,str(second_most_recent_file_t)+"experiment"+ str(exp +1)+ "all_clique_feats.pickle")
            if not os.path.exists(clique_feats_file):
                raise OSError
            with open(clique_feats_file,"rb") as handle:
                all_clique_feats = pickle.load(handle)
            last_processed_t = second_most_recent_file_t
        else:
            obsd_cliques = {} 
            all_clique_feats = {} 
            last_processed_t = 0 
    else:
        obsd_cliques = {} 
        all_clique_feats = {} 
        last_processed_t = 0 

    return obsd_cliques, all_clique_feats, last_processed_t

def read_observations(experiments,results_dir,timesteps=1000,max_feats=1000,serialization_inc=25):    
    '''
    if "empty" in results_dir:
        experiment_name = "empty_map_results"
    else:
        experiment_name = "town01_results"
    ''' 
    err_files = []
    if os.path.exists("debug_feature_detection.txt"): 
        with open("debug_feature_detection.txt","w"):
            #clear this before running again
            pass

    obsd_cliques = {} 
    all_clique_feats = {}
    exp_times = []

    for exp in range(experiments):
        print("this is exp: {}....".format(exp))
        carla_txt_file = os.path.join(results_dir,"gt_car_poses/experiment"+str(exp + 1)+"_gt_car_pose.txt")
        gt_car_traj = get_gt_car_data(carla_txt_file,timesteps)
        data_associations_path = os.path.join(results_dir,"data_association/experiment"+str(exp + 1)+"data_association.csv")
        exp_data_association = np.genfromtxt(data_associations_path)
    
        dir_ = os.path.join(results_dir,"observation_pickles")
        if os.path.exists(os.path.join(dir_,"experiment"+ str(exp +1)+ "observed_cliques.pickle")) and os.path.join(dir_,"experiment"+ str(exp +1)+ "all_clique_feats.pickle"): 
            print("already done with this experiment!")
            continue 

        exp_time0 = time.time()
        exp_observations_count = 0
        all_clique_feats[exp] = {}
        obsd_cliques[exp] = {}

        bb_imgs = [x for x in os.listdir(os.path.join(results_dir,"lm_2dpts")) if x[-3:] == "png"]
        if len(bb_imgs) == 0:
            print(os.path.join(results_dir,"lm_2dpts"))
            for x in os.listdir(os.path.join(results_dir,"lm_2dpts")):
                print("saving bounding box image!")
                rgb_img_path = get_rgb_img_path(results_dir,x)
                save_boundingbox_image(results_dir,rgb_img_path,x)
            bb_imgs = [x for x in os.listdir(os.path.join(results_dir,"lm_2dpts")) if x[-3:] == "png"]

        num_files = len(os.listdir(os.path.join(results_dir,"lm_2dpts")))
        if exp == 0:
            #checking for bounding box images
            for i,x in enumerate(os.listdir(os.path.join(results_dir,"lm_2dpts"))):
                print("done with bounding box images for {} out of {} files".format(i,num_files))
                #print("saving bounding box image!")
                if x[-3:] == "csv":
                    rgb_img_path = get_rgb_img_path(results_dir,x)
                    if rgb_img_path is not None:
                        if not os.path.exists(os.path.join(results_dir,"lm_2dpts/" + x[:-4] + ".png")):
                            try:
                                save_boundingbox_image(results_dir,rgb_img_path,x) 
                            except:
                                print("continuing...")
                                continue 
     
        time_est_increasing = True 

        bb_imgs = [x for x in os.listdir(os.path.join(results_dir,"lm_2dpts")) if x[-3:] == "png"]

        if len(bb_imgs) == 0:
            raise OSError 
        
        bb_imgs_experiments = [x for x in bb_imgs if "experiment"+str(exp+1) in x]
    
        print("there are {} images for this experiment".format(len(bb_imgs_experiments)))

        if len(bb_imgs_experiments) == 0:
            continue 

        tstep_times = []
        previous_estimate = None 

        exp_tree_ids = parse_ids(results_dir,exp + 1,"tree")
        exp_cone_ids = parse_ids(results_dir,exp + 1,"cone")

        last_tstep_w_observations = 0 

        print("Done! Starting to go through the timesteps :)")
        #get 2nd most recent version of this 
        try:
            obsd_cliques, all_clique_feats, last_processed_t = get_most_recent("intermediate_pickles",exp)
        except: 
            obsd_cliques, all_clique_feats, last_processed_t = get_second_most_recent("intermediate_pickles",exp)

        #last_processed_t = 0 

        for t in range(timesteps): 
            if t < last_processed_t: 
                print("we already processed this timestep: {}... moving on".format(t))
                continue 
            '''
            completed_timesteps = t + exp*timesteps; total_tsteps = 12*timesteps 
            percent_done = completed_timesteps / total_tsteps
            print("{} percent done!!!!!".format(100*np.round(percent_done,3)))
            '''
            tstep0 = time.time()

            if t > 100 and exp_observations_count == 0 and last_processed_t == 0:
                print("WARNING: there doesnt seem to be any observations...")
                #raise OSError

            print("parsing experiment {} .... this is t: {}".format(exp,t))
            p = 100*(t / timesteps)
            print("we are {} percent done with this experiment!".format(np.round(p,2)))
            #print("looking for bb imgs from this experiment containing: ", "frame"+str(t).zfill(4))
            bb_imgs_t = [x for x in bb_imgs_experiments if "frame"+str(t).zfill(4) in x]
            filenames = [x[:-3] for x in bb_imgs_t]

            #UNCOMMENT THIS SECTION IF YOU DONT WANT TO LOAD PREVIOUS OBSERVATIONS
            
            if len(bb_imgs_t) > 0:
                observations = {}
                #print("these are the bb files for this timestep: ",[os.path.join(results_dir,"lm_2dpts/"+x+"png") for x in filenames])
                for file in filenames:
                    full_path = os.path.join(results_dir,"lm_2dpts/"+file+"png")
                    print("getting associated features for file: ",full_path)

                    img_filepath = os.path.join(results_dir,"rgb_images/"+"experiment"+ str(exp+1) + "_" + str(t).zfill(4) + "_0.png")

                    #need to avoid doing analgous lm ids 
                    e_idx = [i for i,x in enumerate(file) if x == "e"][-1]
                    lm_id = int(file[e_idx+1:-1])

                    if "tree" in file:
                        if lm_id not in exp_tree_ids:
                            #print("this: {} is an analagous id... moving on".format(lm_id))
                            continue 
                    else:   
                        if lm_id not in exp_cone_ids:
                            #print("this: {} is an analagous id... moving on".format(lm_id))
                            continue 

                    if not os.path.exists(img_filepath):
                        print("this doesnt exist: ",img_filepath)
                        raise OSError 

                    observation_coords_path = os.path.join(results_dir,"lm_2dpts",file + "csv")
                    #print("observation_coords_path: ",observation_coords_path)

                    if "tree" in file:
                        idx = file.index("tree")
                    elif "cone" in file:
                        idx = file.index("cone")

                    #print("file: ",file)
                    #print("file[idx+4:-4]: ",file[idx+4:-1])
                    lm_id = int(file[idx+4:-1])

                    bb_coords = np.genfromtxt(observation_coords_path,delimiter="")

                    x_min = bb_coords[0]; y_min = bb_coords[1]
                    x_max = bb_coords[2]; y_max = bb_coords[3]

                    if np.isnan([x_min,x_max,y_min,y_max]).any():
                        raise OSError 

                    is_n = 50 + exp + 1 

                    if not os.path.exists(os.path.join(results_dir,"is_images","experiment"+str(exp +1)+"_"+str(t).zfill(4)+"_"+str(is_n)+".png")):
                        print("is_n:",is_n)
                        print("this does not exist:",os.path.join(results_dir,"is_images","experiment"+str(exp +1)+"_"+str(t).zfill(4)+"_"+str(is_n)+".png"))
                        raise OSError 

                    is_img = cv2.imread(os.path.join(results_dir,"is_images","experiment"+str(exp +1)+"_"+str(t).zfill(4)+"_"+str(is_n)+".png")); is_img = cv2.cvtColor(is_img, cv2.COLOR_BGR2RGB)
                    is_img_filepath = os.path.join(results_dir,"is_images","experiment"+str(exp +1)+"_"+str(t).zfill(4)+"_"+str(is_n)+".png")
                    #print("is_img_filepath: ",is_img_filepath)

                    if not os.path.exists(os.path.join(results_dir,"depth_images","experiment"+str(exp +1)+"_"+str(t).zfill(4)+"_20.png")): 
                        print("this does not exist: ",os.path.join(results_dir,"depth_images","experiment"+str(exp +1)+str(t).zfill(4)+"_20.png"))
                        raise OSError 

                    #depth_img_path = os.path.join(results_dir,"depth_images","experiment"+str(exp +1)+"_"+str(t).zfill(4)+"_20.png")
                    #depth_img = cv2.imread(os.path.join(results_dir,"depth_images","experiment"+str(exp +1)+"_"+str(t).zfill(4)+"_20.png"))
                    #file_name = "experiment"+str(exp +1)+"_"+str(t).zfill(4)

                    if not os.path.exists(os.path.join(results_dir,"orb_feats/orb_kps",file + "csv")):
                        #print("{} doesnt exist!".format(os.path.join(results_dir,"orb_feats/orb_kps",file + ".csv")))
                        #print("extracting orb features...")
                        extract_orb_feats(results_dir,img_filepath,os.path.join(results_dir,"lm_2dpts",file + "png"))

                    if not os.path.exists(os.path.join(results_dir,"orb_feats/orb_des",file + "csv")):
                        extract_orb_feats(results_dir,img_filepath,os.path.join(results_dir,"lm_2dpts",file + "png")) 
                        
                    #extract_orb_feats(results_dir,img_filepath,os.path.join(results_dir,"lm_2dpts",file + "png"))
                    #img_filepath = os.path.join(results_dir,"lm_2dpts",file + "png")
                    orb_pts = np.genfromtxt(os.path.join(results_dir,"orb_feats/orb_kps",file + "csv"))
                    orb_des = np.genfromtxt(os.path.join(results_dir,"orb_feats/orb_des",file + "csv"))

                    plt.imshow(cv2.imread(img_filepath))
                    plt.scatter(orb_pts[:,0],orb_pts[:,1],color="r",s=2)
                    orb_file = os.path.splitext(os.path.basename(img_filepath))[0]
                    #print("writing the orb feature image for debugging: ",os.path.join(results_dir,"orb_feats/orb_kps",orb_file + ".png"))
                    plt.savefig(os.path.join(results_dir,"orb_feats/orb_kps",orb_file + ".png"))
                    plt.close()

                    #filter patches of the correct semantic label 
                    if "tree" in file: 
                        lm_type = 9
                    else:
                        lm_type = 20 

                    #print("getting associated features...")
                    #print("this is the is_img filepath: ",is_img_filepath)

                    patch_coords = get_associated_features(img_filepath,orb_pts,is_img,lm_id,lm_type,(x_min,y_min,x_max,y_max),file,gt_car_traj[t,:],exp_data_association)

                    observations[lm_id] = {} 
                    for i,orb_feat in enumerate(patch_coords):                      
                        if lm_id in all_clique_feats.keys():
                            #weve already initted this landmark 
                            if len(all_clique_feats[lm_id].keys()) == 0:
                                feat_des_lm_id = []
                            else:
                                feat_des_lm_id = [] 
                                for x in all_clique_feats[lm_id].keys():
                                    feat_des_x = all_clique_feats[lm_id][x]["feat_des"]
                                    feat_des_lm_id.append(feat_des_x)
                            match_found = any(np.array_equal(orb_des[i,:], arr) for arr in feat_des_lm_id)
                            #print("checking for redundant features...")experiment0frame0007cone317.csv
                            if len(feat_des_lm_id) == 0 or not match_found:
                                #add new feature
                                if len(all_clique_feats[lm_id].keys()) == 0:
                                    feat_id = 0 
                                else:
                                    feat_id = max(all_clique_feats[lm_id].keys()) + 1
                                all_clique_feats[lm_id][feat_id] = {}
                                all_clique_feats[lm_id][feat_id]["feat_des"] = orb_des[i,:]
                                all_clique_feats[lm_id][feat_id]["feat_loc"] = orb_feat
                            else:
                                feat_id = [idx for idx, arr in enumerate(feat_des_lm_id) if np.array_equal(orb_des[i, :], arr)][0]
                                if feat_id > max_feats:
                                    continue 
                            observations[lm_id][feat_id] = {} 
                            observations[lm_id][feat_id]["feat_des"] = orb_des[i,:]
                            observations[lm_id][feat_id]["feat_loc"] = orb_feat
                        else:
                            #init a new landmark 
                            all_clique_feats[lm_id] = {}
                            all_clique_feats[lm_id][0] = {}
                            all_clique_feats[lm_id][0]["feat_des"] = orb_des[i,:]
                            all_clique_feats[lm_id][0]["feat_loc"] = orb_feat
                            observations[lm_id][0] = {} 
                            observations[lm_id][0]["feat_des"] = orb_des[i,:]
                            observations[lm_id][0]["feat_loc"] = orb_feat
            
                del_ids = [x for x in observations.keys() if len(observations[x].keys()) == 0]
                for lm_id in del_ids:
                    del observations[lm_id]
                print("observations:",observations)
                print("there are {} observations at this timestep".format(len(observations)))
                print("last_tstep_w_observations: ",last_tstep_w_observations)
                if len(observations) == 0:
                    t_since_last_observation = t - last_tstep_w_observations 
                    if t_since_last_observation < 0:
                        print("t: ",t)
                        print("last_processed_t: ",last_processed_t)
                        print("last_tstep_w_observations: ",last_tstep_w_observations)
                        raise OSError
                    '''
                    if last_processed_t == 0:
                        print("its been {} timesteps since the last observation!".format(t_since_last_observation))
                    else:
                        t_since_last_observation -= last_processed_t
                        print("its been {} timesteps since the last observation!".format(t_since_last_observation)) 
                        print("observations: ",observations)
                        if t_since_last_observation < 0: 
                            raise OSError 
                    '''
                    print("its been {} timesteps since the last observation!".format(t_since_last_observation))
                    if t_since_last_observation > 100:
                        err_files.append(full_path)
                else:
                    last_tstep_w_observations = t 

                for lm_id in observations.keys(): 
                    obsd_features = len(observations[lm_id].keys())
                    #print("observed {} features for clique {} in this timestep".format(obsd_features,lm_id))
                if exp not in obsd_cliques.keys():
                    obsd_cliques[exp] = {} 
                obsd_cliques[exp][t] = observations
                exp_observations_count += len(observations)
                
            else:
                if exp not in obsd_cliques.keys():
                    obsd_cliques[exp] = {} 
                print("obsd_clique[exp][t] is empty!!")
                obsd_cliques[exp][t] = {}
                
            '''
            if t - last_tstep_w_observations > 500: 
                print("observations: ",observations)
                if last_processed_t == 0:
                    print("difference: ",t - last_tstep_w_observations)
                    print("last_processed_t: ",last_processed_t)
                    print("t: ",t)
                    print("last_tstep_w_observations: ",t)
                    with open("debug_feature_detection.txt","w") as handle:
                        for x in err_files:
                            #print("x:",x)
                            handle.write(x + "\n")
                    handle.close()
                    raise OSError 
                else:
                    if t - last_tstep_w_observations - last_processed_t > 500:
                        print("difference: ",t - last_tstep_w_observations - last_processed_t)
                        print("last_processed_t: ",last_processed_t)
                        print("t: ",t)
                        print("last_tstep_w_observations: ",t)
                        with open("debug_feature_detection.txt","w") as handle:
                            for x in err_files:
                                #print("x:",x)
                                handle.write(full_path + "\n")
                        handle.close()
                        raise OSError
            '''
            
            tstep1 = time.time()
            tstep_times.append(tstep1-tstep0)
            #print("mean finishing time: ",np.round(np.mean(tstep_times),1))
            percent_done = np.round(100*(t / timesteps),2)
            print("{} percent done with this experiment!".format(percent_done))
            if len(tstep_times) > 10:
                remaining_tsteps = timesteps - t
                mean_finishing_time_secs = np.mean(tstep_times)
                remaining_secs = remaining_tsteps * mean_finishing_time_secs  
                if previous_estimate is None:
                    previous_estimate = np.round(remaining_secs/60,3)
                else:
                    if previous_estimate < np.round(remaining_secs/60,3) and time_est_increasing and t > 100:
                        print("estimate is still increasing...cant estimate finshing time rn...best_guess is {} minutes left for this experiment".format(np.round(remaining_secs/60,3)))
                    else:
                        time_est_increasing = False 
                        print("there is about {} min left for this experiment!".format(np.round(remaining_secs/60,3)))
                    if remaining_secs < 0:
                        print("remaining_secs: ",remaining_secs)
                        print("remaining_tsteps: ",remaining_tsteps)
                        print("np.mean(tstep_times): ",np.mean(tstep_times))
                        raise OSError
                '''
                 else:
                    mean_tstep_secs = np.mean(exp_times)/timesteps
                    x = remaining_tsteps*mean_tstep_secs/60
                    print("there should be about {} minutes remaining in this experiment".format(np.round(x,3)))
                '''
               
                if experiments > 1:
                    remaining_experiments = experiments - exp - 1 
                    total_remaining_tsteps = remaining_experiments * timesteps + remaining_tsteps
                    x = total_remaining_tsteps * mean_finishing_time_secs / 60**2
                    if len(tstep_times) > 100:
                        print("maybe {} hours left to do everthing".format(np.round(x,3)))

            if np.mod(t,serialization_inc) == 0 and t > 0: 
                print("writing intermediate pickles...")
                if not os.path.exists("intermediate_pickles"):
                    os.mkdir("intermediate_pickles")
                #serialize experiment0frame0007cone317.csv
                with open(os.path.join("intermediate_pickles",str(t) + "experiment"+ str(exp +1)+ "observed_cliques.pickle"),"wb") as handle:
                    pickle.dump(obsd_cliques,handle,protocol=pickle.HIGHEST_PROTOCOL)

                with open(os.path.join("intermediate_pickles",str(t) + "experiment"+ str(exp +1)+ "all_clique_feats.pickle"),"wb") as handle:
                    pickle.dump(all_clique_feats,handle,protocol=pickle.HIGHEST_PROTOCOL)

        #serialize 
        print("serialize......")
        if not os.path.exists(os.path.join(results_dir,"observation_pickles")):
            os.mkdir(os.path.join(results_dir,"observation_pickles"))

        dir_ = os.path.join(results_dir,"observation_pickles")
        print("writing: ",os.path.join(dir_,"experiment"+ str(exp +1)+ "observed_cliques.pickle"))
        with open(os.path.join(dir_,"experiment"+ str(exp +1)+ "observed_cliques.pickle"),"wb") as handle:
            pickle.dump(obsd_cliques,handle,protocol=pickle.HIGHEST_PROTOCOL)
        print("writing: ",os.path.join(dir_,"experiment"+ str(exp +1)+ "all_clique_feats.pickle"))
        with open(os.path.join(dir_,"experiment"+ str(exp +1)+ "all_clique_feats.pickle"),"wb") as handle:
            pickle.dump(all_clique_feats,handle,protocol=pickle.HIGHEST_PROTOCOL)

        exp_time = time.time() - exp_time0
        exp_times.append(exp_time)
        print("finished experiment {}... that took {} minutes".format(exp,np.round(exp_time/60,3)))

    return obsd_cliques,all_clique_feats

def main(args):
    #these are the parameters of the camera we used in carla to get the orb features :)
    width = 1392; height = 1024; fov = 72 
    camera_offset = [0.3,0,1.7] 
    #map_name = args.map_name
    map_name = "empty_map_results"

    max_dist = args.max_dist
    experiments = args.experiments
    #sensor_noise_variance = args.sensor_noise_variance 

    '''
    if os.path.exists("/media/arpg/easystore1/BetterFaster2.0/run_carla_experiments"):
        results_dir = os.path.join("/media/arpg/easystore1/BetterFaster2.0/run_carla_experiments",map_name)
    else: 
        if os.path.exists(os.path.join("/media/arpg/easystore/BetterFaster2.0/run_carla_experiments",map_name)):
            results_dir = os.path.join("/media/arpg/easystore/BetterFaster2.0/run_carla_experiments",map_name) 
        else:
            print("Results path doesnt exist!")
            raise OSError 
    '''

    if not os.path.exists("/media/arpg/easystore1/BetterFaster/kitti_carla_simulator/results"):
        results_dir = "/media/arpg/easystore/BetterFaster/kitti_carla_simulator/exp_results"
    else:
        results_dir = "/media/arpg/easystore1/BetterFaster/kitti_carla_simulator/exp_results"

    obsd_cliques,all_cliques_feats = read_observations(experiments,results_dir,timesteps=int(args.timesteps))

if __name__ == "__main__":
    #this function goes through all the frames in the experiment to find the clique features and associate them to a
    #timestep
    parser = argparse.ArgumentParser()

    # Add sim_length argument
    #parser.add_argument('-map_name', type=str, help='Map name. Either: ["empty_map_results","town01_results"].')
    parser.add_argument('--max_dist',type=float,default=100,help="Optional parameter: farthest range which a landmark is detectable")
    parser.add_argument('--experiments',type=int,default=12)
    parser.add_argument('--sensor_noise_variance',type=float,default=0.1)
    parser.add_argument('--timesteps',type=int,default=1000)

    args = parser.parse_args()
    #print("these are the args:",args)
    main(args)