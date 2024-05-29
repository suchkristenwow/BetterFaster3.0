import numpy as np
import matplotlib.pyplot as plt 
import random 
import pickle 
from utils import point_in_trapezoid
import math 
import os 

'''
detx["clique_id"] = lm_id 
detx["feature_id"] = feat_id
detx["bearing"] = bearing 
detx["range"] = range_
detx["detection"] = bearing #binary random variable 
'''

def get_gt_lms_feats(gt_lms,exp,persistence):
    #gt landmarks 
    #covariance_matrix = np.array([[0.25, 0.5], [0.5, 1]])  
    covariance_matrix = np.random.rand(2, 2)
    #gt_lms = [[10,5],[-10,3],[-10,-5],[10,-5]]
    all_clique_feats = {} 

    for i,arr in enumerate(gt_lms):
        lm = arr[1:]
        id_ = arr[0]
        if persistence[i]:
            all_clique_feats[id_] = {}
            n_feats = random.randint(3,100)
            feat_locations = np.random.multivariate_normal(np.array(lm), covariance_matrix, n_feats)
            for j in range(n_feats): 
                all_clique_feats[id_][j] = {} 
                all_clique_feats[id_][j]["feat_loc"] = feat_locations[j,:]
        else:
            print("lm id:{} does not persist at this experiment: {}".format(id_,exp))
    file_path = "exp"+str(exp)+"all_clique_feats.pickle"
    with open(os.path.join("/home/kristen/BetterFaster3.0/sim_utils/fake_data",file_path),"wb") as handle:
        pickle.dump(all_clique_feats,handle)

def get_gt_traj():
    #way points 
    gt_way_pts = [[8,0,90],[8,12,180],[-3,12,270],[-3,0,180],[-12,0,270],[-12,-12,0],[8,-12,90]]

    tsteps_loop = 500 #500 tsteps/loop, 10 loops 

    #plt.ion() 
    #fig,ax = plt.subplots() 
    results_dir = "/home/kristen/BetterFaster3.0/sim_utils/fake_data"

    AB_y = np.linspace(0, 12, 80)
    BC_x = np.linspace(8, -3, 75)
    CD_y = np.linspace(12, 0, 70)
    DE_x = np.linspace(-3, -12, 70)
    EF_y = np.linspace(0, -12, 30)
    FG_x = np.linspace(-12, 8, 41)
    GA_y = np.linspace(-12, 0, 10)

    gt_traj = np.zeros((tsteps_loop,3))

    for i in range(tsteps_loop): 
        pose = np.zeros((3,))
        if 0 <= i < tsteps_loop*.02:
            #way point A 
            pose = gt_way_pts[0]
        elif tsteps_loop*.02<= i < tsteps_loop*0.18: 
            #AB 
            pose[0] = 8 
            pose[2] = 90
            idx = i-int(tsteps_loop*.02)
            pose[1] = AB_y[idx]
        elif tsteps_loop*0.18 <= i < tsteps_loop*0.30: 
            #way point B
            pose = gt_way_pts[1]
        elif tsteps_loop*0.30 <= i < tsteps_loop*0.45:
            #BC 
            pose[1] = 12 
            pose[2] = 180
            idx = i- int(tsteps_loop*0.30)
            pose[0] = BC_x[idx]
        elif tsteps_loop*0.45 <= i < tsteps_loop*0.5: 
            #way point C
            pose = gt_way_pts[2]
        elif tsteps_loop*0.5 <= i < tsteps_loop*0.64:
            #CD
            pose[0] = -3 
            pose[2] = 270
            idx = i - int(tsteps_loop*0.5)
            pose[1] = CD_y[idx]
        elif  tsteps_loop*0.64 <= i < tsteps_loop*0.7:
            # way point D
            pose = gt_way_pts[3]
        elif tsteps_loop*0.7 <= i < tsteps_loop*0.84:
            #DE
            pose[1] = 0 
            pose[2] = 180
            idx = i-int(tsteps_loop*0.7)
            pose[0] = DE_x[idx]
        elif i == tsteps_loop*0.84:
            #way point E
            pose = gt_way_pts[4]
        elif tsteps_loop*0.84 < i < tsteps_loop*0.9:
            #EF
            pose[0] = -12 
            pose[2] = 270
            idx = i - int(tsteps_loop*0.84)
            pose[1] = EF_y[idx]
        elif i == int(tsteps_loop*0.9):
            #way point F
            pose = gt_way_pts[5]
        elif tsteps_loop*0.9 < i <= tsteps_loop*0.98:
            #FG 
            pose[1] = -12 
            pose[2] = 0
            idx = i - int(tsteps_loop*0.9)
            pose[0] = FG_x[idx]
        elif int(tsteps_loop*0.98) == i:
            #way point G 
            pose = gt_way_pts[6]
        elif tsteps_loop*0.98 < i < tsteps_loop:
            #GA
            pose[0] = 8
            pose[2] = 90
            idx =  i - int(tsteps_loop*0.98)
            pose[1] = GA_y[idx]

        gt_traj[i,:] = pose 
        '''
        ax.clear()
        ax.set_xlim(-15,15)
        ax.set_ylim(-15,15)
        ax.set_aspect('equal')
        print("pose: ",pose)
        ax.scatter(pose[0],pose[1],color="red")
        pointer_x = pose[0] + np.cos(pose[2]*(np.pi/180))
        pointer_y = pose[1] + np.sin(pose[2]*(np.pi/180))
        ax.plot([pose[0],pointer_x],[pose[1],pointer_y],color="red")
        plt.pause(0.05)
        '''
            
    np.savetxt(os.path.join(results_dir,"fake_gt_traj.csv"),gt_traj,delimiter=",")

def get_range_bearing(p0,p1):
    # Extract x, y coordinates from p0 and p1
    x0, y0 = p0[0], p0[1] #camera pose
    '''
    if enable_noise_variance:
        x1, y1 = p1[0], p1[1] #landmark pos
        noisy_landmark = p1 + np.random.normal(0,self.sensor_noise_variance,size=2)
        x1 = noisy_landmark[0]; y1 = noisy_landmark[1]
    else:
    '''
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
    return range_, yaw 

#random_feat_ids = get_obsd_feat_ids(list(all_clique_feats[lm[0]].keys()),all_observations[t-1],subset_size)
def get_obsd_feat_ids(clique_id,clique_feat_ids,previous_observations,n_feats):
    if clique_id in [x["clique_id"] for x in previous_observations]: 
        prev_clique_observations = [x for x in previous_observations if x["clique_id"] == clique_id]
        #pick random subset of features to repeat 
        random_ratio = random.uniform(0.1, 1)
        print("random ratio of repeated features: ",random_ratio)
        n_repeat_feats = int(random_ratio*n_feats)
        n_unique_feats = n_feats - n_repeat_feats  
        if n_repeat_feats > len(prev_clique_observations): 
            repeat_feat_ids = [x["feature_id"] for x in prev_clique_observations] 
            n_repeat_feats = len(repeat_feat_ids)
            n_unique_feats = n_feats - n_repeat_feats            
        else: 
            repeat_feat_ids = random.sample([x["feature_id"] for x in prev_clique_observations],n_repeat_feats)
        available_unique_feats = [x for x in clique_feat_ids if x not in repeat_feat_ids]
        new_feat_ids = random.sample(available_unique_feats,n_unique_feats) 
        #print("new_feat_ids: {}, repeat_feat_ids: {}".format(new_feat_ids,repeat_feat_ids))
        feat_ids = new_feat_ids + repeat_feat_ids
    else: 
        #print("random ratio of repeated features: ",0)
        feat_ids = random.sample(clique_feat_ids,n_feats)
    #print("feat_ids: ",feat_ids)
    return feat_ids 

def get_observations(gt_lms,exp_persistence,exp):
    print("getting observations... this is exp:",exp)
    plt.ion() 
    fig, ax = plt.subplots()
    #literally the trajectory is the same every time 
    gt_traj = np.genfromtxt("fake_data/fake_gt_traj.csv",delimiter=",")
    results_dir = "/home/kristen/BetterFaster3.0/sim_utils/fake_data"
    file_path = "exp"+str(exp)+"all_clique_feats.pickle"
    with open(os.path.join(results_dir,file_path),"rb") as handle:
        all_clique_feats = pickle.load(handle)

    min_d = 0.1; max_d = 10

    all_observations = {} 
    nonzero_observations = 0
    for t in range(len(gt_traj)):
        #print("t:",t)        
        ax.set_xlim([-15,15])
        ax.set_ylim([-15,15]) 
        ax.set_aspect('equal')
        ax.set_title('Experiment {}'.format(exp)) 

        all_observations[t] = []

        robot_pose = gt_traj[t,:]
        #print("robot_pose: ",robot_pose)
        min_theta = np.deg2rad(robot_pose[2]- (72/2))
        max_theta = np.deg2rad(robot_pose[2] + (72/2))

        x0 = robot_pose[0] + min_d*np.cos(min_theta)
        y0 = robot_pose[1] + min_d*np.sin(min_theta)
        x1 = robot_pose[0] + min_d*np.cos(max_theta)
        y1 = robot_pose[1] + min_d*np.sin(max_theta)
        x2 = robot_pose[0] + max_d*np.cos(min_theta)
        y2 = robot_pose[1] + max_d*np.sin(min_theta)
        x3 = robot_pose[0] + max_d*np.cos(max_theta)
        y3 = robot_pose[1] + max_d*np.sin(max_theta)
        verts = [(x0,y0),(x1,y1),(x2,y2),(x3,y3)] #these are the verts of the frustrum 

        miss_detection_probability_function = lambda d: -2*10**(-3)*d**2 + .185*d 
        observations_t = []

        for i,lm in enumerate(gt_lms):
            if len(all_clique_feats.keys()) == 0:
                raise OSError
            else:
                print("all_clique_feats.keys():",all_clique_feats.keys())
            print("lm:",lm)
            if lm[0] in all_clique_feats.keys():
                #random_feat_ids = list(all_clique_feats.keys())
                print("all_clique_feats[lm[0]]:",all_clique_feats[lm[0]].keys())
                subset_size = random.randint(0, len(all_clique_feats[lm[0]].keys()))
                if 1  <= t:
                    random_feat_ids = get_obsd_feat_ids(lm[0],list(all_clique_feats[lm[0]].keys()),all_observations[t-1],subset_size)
                else:
                    random_feat_ids = random.sample(list(all_clique_feats[lm[0]].keys()), subset_size)
                print("random_feat_ids: ",random_feat_ids)
                print("lm[1:]: ",lm[1:])
                if exp_persistence[i]:
                    if point_in_trapezoid(lm[1:],verts):
                        #print("this lm is in the frustrum!")
                        for id_ in all_clique_feats[lm[0]].keys():
                            if id_ in random_feat_ids:
                                feat_loc = all_clique_feats[lm[0]][id_]["feat_loc"]
                                #get range and bearing 
                                r,b = get_range_bearing(robot_pose,feat_loc)
                                if point_in_trapezoid(all_clique_feats[lm[0]][id_]["feat_loc"],verts):
                                    detx = {}    
                                    detx["clique_id"] = lm[0]
                                    detx["feature_id"] = id_ 
                                    detx["range"] = r 
                                    detx["bearing"] = b
                                    if miss_detection_probability_function(r) > np.random.random():
                                        detx["detection"] = 1  
                                        nonzero_observations += 1 
                                    else:
                                        detx["detection"] = 0
                                    observations_t.append(detx) 
        
        ax.scatter(robot_pose[0],robot_pose[1],color="red")
        #robot pointer 
        ax.plot([robot_pose[0],robot_pose[0] + np.cos((np.pi/180)*robot_pose[2])],[robot_pose[1],robot_pose[1] + np.sin((np.pi/180)*robot_pose[2])],color="red") #pointer
        # plot the frustrum 
        ax.plot([x1,x0],[y1,y0],'b')
        ax.plot([x0,x2],[y0,y2],'b')
        ax.plot([x2,x3],[y2,y3],'b')
        ax.plot([x3,x1],[y3,y1],'b')
        
        for i in range(len(gt_lms)): 
            print("exp_persistence: ",exp_persistence)
            print("exp_persistence[i]: ",exp_persistence[i])
            if exp_persistence[i]:
                ax.scatter(gt_lms[i,1],gt_lms[i,2],color="red",marker="*")
                #ax.scatter([x[1] for x in gt_lms],[x[2] for x in gt_lms],color="red",marker="*") 
            else:
                ax.scatter(gt_lms[i,1],gt_lms[i,2],color="k",marker="*")
                #ax.scatter([x[1] for x in gt_lms],[x[2] for x in gt_lms],color="k",marker="*") 

        for obs in observations_t:
            if obs["detection"]:
                observation_x = robot_pose[0] + obs["range"]*np.cos((np.pi/180)*obs["bearing"])
                observation_y = robot_pose[1] + obs["range"]*np.sin((np.pi/180)*obs["bearing"])
                ax.plot([robot_pose[0],observation_x],[robot_pose[1],observation_y],linestyle="-.",color="red")

        plt.pause(0.025)
        ax.clear()

        '''
        if exp > 2:
            ax.scatter(robot_pose[0],robot_pose[1],color="red")
            #robot pointer 
            ax.plot([robot_pose[0],robot_pose[0] + np.cos((np.pi/180)*robot_pose[2])],[robot_pose[1],robot_pose[1] + np.sin((np.pi/180)*robot_pose[2])],color="red") #pointer
            # plot the frustrum 
            ax.plot([x1,x0],[y1,y0],'b')
            ax.plot([x0,x2],[y0,y2],'b')
            ax.plot([x2,x3],[y2,y3],'b')
            ax.plot([x3,x1],[y3,y1],'b')
            
            for i in range(len(gt_lms)): 
                print("exp_persistence: ",exp_persistence)
                print("exp_persistence[i]: ",exp_persistence[i])
                if exp_persistence[i]:
                    ax.scatter(gt_lms[i,1],gt_lms[i,2],color="red",marker="*")
                    #ax.scatter([x[1] for x in gt_lms],[x[2] for x in gt_lms],color="red",marker="*") 
                else:
                    ax.scatter(gt_lms[i,1],gt_lms[i,2],color="k",marker="*")
                    #ax.scatter([x[1] for x in gt_lms],[x[2] for x in gt_lms],color="k",marker="*") 

            for obs in observations_t:
                if obs["detection"]:
                    observation_x = robot_pose[0] + obs["range"]*np.cos((np.pi/180)*obs["bearing"])
                    observation_y = robot_pose[1] + obs["range"]*np.sin((np.pi/180)*obs["bearing"])
                    ax.plot([robot_pose[0],observation_x],[robot_pose[1],observation_y],linestyle="-.",color="red")

            plt.pause(0.025)
            ax.clear()
        '''
        
        #print("this is observations_t: ",observations_t)
        all_observations[t] = observations_t 
    
    #print("writing {}".format("exp" +str(exp)+ "_fake_observations.pickle ...."))
    '''
    file_path = "exp_"+str(exp)+"_fake_all_clique_feats.pickle"
    with open(os.path.join("/home/kristen/BetterFaster3.0/sim_utils/fake_data",file_path),"wb") as handle:
        pickle.dump(all_clique_feats,handle)
    '''
    #file_path = "exp" +str(exp)+ "_fake_observations.pickle" 
    file_path = "exp" + str(exp) + "observed_cliques.pickle"
    results_dir = "/home/kristen/BetterFaster3.0/sim_utils/fake_data"
    with open(os.path.join(results_dir,file_path),"wb") as handle:
        pickle.dump(all_observations,handle)

    plt.close() 


if __name__ == "__main__":
    get_gt_traj() 
    experiments = 10 
    gt_lms = np.array([[0,10,5],[1,-10,3],[2,-10,-5],[3,10,-5],[4,6,3],[5,-8,12],[6,0,-11]])
    tree_ids = [0,1,2,3]
    cone_ids = [4,5,6] 

    persistence = np.ones((10,7))
    persistence[3:,0] = 0 
    persistence[4:,1] = 0
    persistence[7:,4] = 0

    random_array = [10,20,30,40,50,60,70,80,90,100]

    results_dir = "/home/kristen/BetterFaster3.1/betterFaster/sim_utils/fake_data"
    if not os.path.exists(os.path.join(results_dir,"cone_ids")):
        os.mkdir(os.path.join(results_dir,"cone_ids"))

    if not os.path.exists(os.path.join(results_dir,"tree_ids")):
        os.mkdir(os.path.join(results_dir,"tree_ids"))

    if not os.path.exists(os.path.join(results_dir,"data_associations")):
        os.mkdir(os.path.join(results_dir,"data_associations"))

    for exp in range(experiments):
        #print("this is exp: ",exp)
        if exp > 0:
            gt_lm_copy = gt_lms.copy()
            #print("gt_lm_copy.shape: ",gt_lm_copy.shape)
            for i in range(gt_lms.shape[0]): 
                if persistence[exp,i]:
                    idx = np.where(gt_lm_copy[:,0] == i)
                    gt_lm_copy[idx,0] = random_array[exp] + i  
                    #print("changed the gt lm array id to: ",random_array[exp] + i)
                    if i in tree_ids:
                        with open(os.path.join(results_dir,"tree_ids/experiment"+str(exp+1)+"tree_ids_.txt"),"a") as f:
                            f.write(str(random_array[exp] + i) + "\n") 
                        f.close() 
                    else:
                        with open(os.path.join(results_dir,"cone_ids/experiment"+str(exp+1)+"cone_ids_.txt"),"a") as f:
                            f.write(str(random_array[exp] + i) + "\n") 
                        f.close() 
                else:                
                    '''
                    print("This lm does not persist")
                    print("this is i: ",i)
                    print("gt_lms[:,0]: ",gt_lms[:,0])
                    '''
                    idx = np.where(gt_lms[:,0] == i)
                    #print("idx: ",idx)
                    gt_lm_copy = np.delete(gt_lm_copy, idx, axis=0)
                    #print("gt_lm_copy:",gt_lm_copy)
                    #print("this is gt_lms: ",gt_lms)
            if gt_lms.shape[0] != 7:
                raise OSError
            path = os.path.join(results_dir,"data_associations/exp" +str(exp)+"_data_association.csv")
            np.savetxt(path,gt_lm_copy)
        else:
            #this is the first experiment 
            print("gt_lms:",gt_lms)
            for i in range(gt_lms.shape[0]):
                if i in cone_ids:
                    with open(os.path.join(results_dir,"cone_ids/experiment"+str(exp+1)+"cone_ids_.txt"),"a") as f:
                        f.write(str(i) + "\n") 
                    f.close() 
                else: 
                    with open(os.path.join(results_dir,"tree_ids/experiment"+str(exp+1)+"tree_ids_.txt"),"a") as f:
                        f.write(str(i) + "\n") 
                    f.close() 

            path = os.path.join(results_dir,"data_associations/exp" +str(exp)+"_data_association.csv")
            np.savetxt(path,gt_lms)

        get_gt_lms_feats(gt_lms,exp,persistence[exp,:]) 
        get_observations(gt_lms,persistence[exp,:],exp) 