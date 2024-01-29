import os 
import argparse
import time 
import numpy as np
import cv2
from scipy import stats as st
import pickle 
import matplotlib.pyplot as plt 

def debug_image_patches(img_path,coords):
    img = cv2.imread(img_path)
    plt.imshow(img)
    for coord in coords:
        plt.scatter(coord[0],coord[1])
    plt.show() 

def get_obscured_image_patch(img_path,orb_feat_coords,is_img,depth_img):
    patch_coords = []
    depth_map = []
    '''
    for i in range(int(x_min),int(x_max)):
        for j in range(int(y_min),int(y_max)):
    '''
    for orb_feat_coord in orb_feat_coords:
        j = int(orb_feat_coord[0]); i = int(orb_feat_coord[1])
        px = is_img[i,j]
        if px[2] in [9,20]: #is it a traffic cone or a tree?
            patch_coords.append([i,j])
        px = depth_img[i,j]
        B = px[0]; G = px[1]; R = px[2]
        depth_map.append(R + G*256 + B*256*256)

    #remove occlusions
    mode_depth = st.mode(depth_map).mode[0]
    #print("mode_depth: ",mode_depth)
    purge_idx = []
    for i,x in enumerate(patch_coords):
        px = depth_img[x[0],x[1]]
        B = px[0]; G = px[1]; R = px[2]
        #print("R + G*256 + B*256*256:",R + G*256 + B*256*256)
        if R + G*256 + B*256*256 != mode_depth:
            purge_idx.append(i)

    orig_patch_coords = patch_coords
    orig_patch_coords_length = len(patch_coords)
    patch_coords = [patch_coords[i] for i in range(len(patch_coords)) if i not in purge_idx]
    if len(patch_coords) == 0 and len(purge_idx) == orig_patch_coords_length:
        if orig_patch_coords_length > 0:
            debug_image_patches(img_path,orig_patch_coords)
            raise OSError   
    return patch_coords

def read_observations(experiments,results_dir,orb_dir):    
    if "empty" in results_dir:
        experiment_name = "empty_map_results"
    else:
        experiment_name = "town01_results"
        
    obsd_cliques = {} 
    all_clique_feats = {}
    exp_times = []

    for exp in range(experiments):
        exp_time0 = time.time()
        exp_observations_count = 0
        all_clique_feats[exp] = {}
        obsd_cliques[exp] = {}
        bb_imgs = [x for x in os.listdir(os.path.join(results_dir,"lm_2dpts")) if x[-3:] == "png"]
        if len(bb_imgs) == 0:
            raise OSError 
        bb_imgs_experiments = [x for x in bb_imgs if "experiment"+str(exp) in x]
        #print("there are {} images for this experiment".format(len(bb_imgs_experiments)))
        if len(bb_imgs_experiments) == 0:
            raise OSError 
        tstep_times = []
        previous_estimate = None 
        for t in range(500):
            completed_timesteps = t + exp*500; total_tsteps = 12*500 
            percent_done = completed_timesteps / total_tsteps
            print("{} percent done!!!!!".format(100*np.round(percent_done,3)))
            tstep0 = time.time()
            if t > 100 and exp_observations_count ==0:
                raise OSError
            print("parsing experiment {} .... this is t: {}".format(exp,t))
            bb_imgs_t = [x for x in bb_imgs_experiments if "frame"+str(t).zfill(4) in x]
            filenames = [x[:-3] for x in bb_imgs_t]
            c = 0
            if len(bb_imgs_t) > 0:
                observations = {}
                #orb_pts = np.genfromtxt(os.path.join(orb_dir,test_file_name + ".csv"),delimiter=",")
                for file in filenames:
                    img_filepath = os.path.join(results_dir,"lm_2dpts",file + "png")
                    #print("img_filepath: ",img_filepath)
                    observation_coords_path = os.path.join(results_dir,"lm_2dpts",file + "csv")
                    if "tree" in file:
                        idx = file.index("tree")
                    elif "cone" in file:
                        idx = file.index("cone")
                    lm_id = int(file[idx+4:-4])
                    bb_coords = np.genfromtxt(observation_coords_path,delimiter=",")
                    coord1 = (bb_coords[0],bb_coords[2])
                    coord2 = (bb_coords[1],bb_coords[3])
                    x_min = min([coord1[0],coord2[0]]); x_max = max([coord1[0],coord2[0]]); 
                    y_min = min([coord1[1],coord2[1]]); y_max = max([coord1[1],coord2[1]])
                    if not os.path.exists(os.path.join(results_dir,"ss_images","experiment"+str(exp)+"_"+str(t).zfill(4)+"_10.png")):
                        print("this does not exist:",os.path.join(results_dir,"ss_images","experiment"+str(exp)+"_"+str(t).zfill(4)+"_10.png"))
                        raise OSError 
                    is_img = cv2.imread(os.path.join(results_dir,"ss_images","experiment"+str(exp)+"_"+str(t).zfill(4)+"_10.png"))

                    if not os.path.exists(os.path.join(results_dir,"depth_images","experiment"+str(exp)+"_"+str(t).zfill(4)+"_20.png")): 
                        print("this does not exist: ",os.path.join(results_dir,"depth_images","experiment"+str(exp)+str(t).zfill(4)+"_20.png"))
                        raise OSError 
                    depth_img = cv2.imread(os.path.join(results_dir,"depth_images","experiment"+str(exp)+"_"+str(t).zfill(4)+"_20.png"))
                    file_name = "experiment"+str(exp)+"_"+str(t).zfill(4)
                    orb_pts = np.genfromtxt(os.path.join(orb_dir,experiment_name + "_orb_feats",file_name + ".csv"),delimiter=",")
                    orb_des = np.genfromtxt(os.path.join(orb_dir,experiment_name + "_orb_des",file_name + ".csv"),delimiter=",")
                    observations_t = []
                    width = x_max - x_min; height = y_max - y_min 
                    patch_coords = get_obscured_image_patch(img_filepath,orb_pts,is_img,depth_img)
                    #for i,orb_feat in enumerate(orb_pts):
                    #if any(np.array_equal(orb_feat, arr) for arr in patch_coords):
                    for i,orb_feat in enumerate(patch_coords):
                        if x_min + 0.1*width <= orb_feat[0] <= x_max - 0.1*width:
                            if y_min + 0.1*height <= orb_feat[1] <= y_max - 0.1*height:
                                #print("observed this feat!")
                                observations_t.append(orb_feat)
                                observations[c] = {}
                                observations[c]["lm_id"] = lm_id
                                observations[c]["feat_des"] = orb_des[i,:]
                                observations[c]["feat_loc"] = orb_feat
                                if lm_id in all_clique_feats.keys():
                                    if len(all_clique_feats[lm_id].keys()) == 0:
                                        feat_des_lm_id = []
                                    else:
                                        feat_des_lm_id = [] 
                                        for x in all_clique_feats[lm_id].keys():
                                            feat_des_x = all_clique_feats[lm_id][x]["feat_des"]
                                            feat_des_lm_id.append(feat_des_x)
                                    match_found = any(np.array_equal(orb_des[i,:], arr) for arr in feat_des_lm_id)
                                    if len(feat_des_lm_id) == 0 or not match_found:
                                        if len(all_clique_feats[lm_id].keys()) == 0:
                                            feat_id = 0 
                                        else:
                                            feat_id = max(all_clique_feats[lm_id].keys()) + 1
                                        all_clique_feats[lm_id][feat_id] = {}
                                        all_clique_feats[lm_id][feat_id]["feat_des"] = orb_des[i,:]
                                        all_clique_feats[lm_id][feat_id]["feat_loc"] = orb_feat
                                else:
                                    all_clique_feats[lm_id] = {}
                                c += 1 
                '''
                img = cv2.imread(img_filepath)
                plt.imshow(img)
                for observed_feat in observations_t:
                    plt.scatter(observed_feat[0],observed_feat[1],color="g")
                plt.show()
                '''
                print("there are {} observations at this timestep".format(len(observations)))
                obsd_cliques[exp][t] = observations
                exp_observations_count += len(observations)
            else:
                print("obsd_clique[exp][t] is empty!!")
                obsd_cliques[exp][t] = {}

            tstep1 = time.time()
            tstep_times.append(tstep1-tstep0)
            if len(tstep_times) > 10:
                remaining_tsteps = 500 - t
                if exp == 0:
                    mean_finishing_time_secs = np.mean(tstep_times)
                    remaining_secs = remaining_tsteps * mean_finishing_time_secs  
                    if previous_estimate is None:
                        previous_estimate = np.round(remaining_secs/60,3)
                    else:
                        if previous_estimate < np.round(remaining_secs/60,3):
                            print("estimate is still increasing...cant estimate finshing time rn")
                        print("there is about {} min left for this experiment!".format(np.round(remaining_secs/60,3)))
                else:
                    mean_tstep_secs = np.mean(exp_times)/500 
                    x = remaining_tsteps*mean_tstep_secs/60
                    print("there should be about {} minutes remaining in this experiment".format(np.round(x,3)))

                remaining_experiments = experiments - exp - 1 
                total_remaining_tsteps = remaining_experiments * 500 + remaining_tsteps
                x = total_remaining_tsteps * mean_finishing_time_secs / 60**2
                if len(tstep_times) > 100:
                    print("maybe {} hours left to do everthing".format(np.round(x,3)))

            if np.mod(t,100) == 0 and t > 0: 
                if not os.path.exists("backup_pickles"):
                    os.mkdir("backup_pickles")
                #serialize 
                with open(os.path.join("intermediate_pickles",str(t) + "experiment"+ str(exp)+ "observed_cliques.pickle"),"wb") as handle:
                    pickle.dump(obsd_cliques,handle,protocol=pickle.HIGHEST_PROTOCOL)

                with open(os.path.join("intermediate_pickles",str(t) + "experiment"+ str(exp)+ str(t) + "all_clique_feats.pickle"),"wb") as handle:
                    pickle.dump(all_clique_feats,handle,protocol=pickle.HIGHEST_PROTOCOL)

        #serialize 
        with open("experiment"+ str(exp)+ "observed_cliques.pickle","wb") as handle:
            pickle.dump(obsd_cliques,handle,protocol=pickle.HIGHEST_PROTOCOL)

        with open("experiment"+ str(exp)+ "all_clique_feats.pickle","wb") as handle:
            pickle.dump(all_clique_feats,handle,protocol=pickle.HIGHEST_PROTOCOL)

        exp_time = time.time() - exp_time0
        exp_times.append(exp_time)
        print("finished experiment {}... that took {} minutes".format(np.round(exp_time/60,3)))

    #serialize 
    with open("observed_cliques.pickle","wb") as handle:
        pickle.dump(obsd_cliques,handle,protocol=pickle.HIGHEST_PROTOCOL)

    with open("all_clique_feats.pickle","wb") as handle:
        pickle.dump(all_clique_feats,handle,protocol=pickle.HIGHEST_PROTOCOL)

    return obsd_cliques,all_clique_feats

def main(args):
    #these are the parameters of the camera we used in carla to get the orb features :)
    width = 1392; height = 1024; fov = 72 
    camera_offset = [0.3,0,1.7] 
    map_name = args.map_name
    max_dist = args.max_dist
    experiments = args.experiments
    sensor_noise_variance = args.sensor_noise_variance 

    results_dir = "/media/kristen/easystore3/BetterFaster2.0/run_carla_experiments/empty_map_results/"
    orb_dir = "/media/kristen/easystore3/ORB_feat_extraction/"

    obsd_cliques,all_cliques_feats = read_observations(experiments,results_dir,orb_dir)

if __name__ == "__main__":
    #this function goes through all the frames in the experiment to find the clique features and associate them to a
    #timestep
    parser = argparse.ArgumentParser()

    # Add sim_length argument
    parser.add_argument('-map_name', type=str, help='Map name. Either: ["empty_map_results","town01_results"].')
    parser.add_argument('--max_dist',type=float,default=100,help="Optional parameter: farthest range which a landmark is detectable")
    parser.add_argument('--experiments',type=int,default=12)
    parser.add_argument('--sensor_noise_variance',type=float,default=0.1)

    args = parser.parse_args()
    print("these are the args:",args)
    main(args)