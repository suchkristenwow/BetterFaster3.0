from JointPersistenceFilter import PersistenceFilter
import random 
import numpy as np 
import sys
sys.path.append('../sim_utils')
from utils import get_reinitted_id
import os 
import pickle 

def last_index_of_change(lst):
    for i in range(len(lst) - 1, 0, -1):
        if lst[i] != lst[i - 1]:
            return i
    return -1  # If there is no change

def get_exp_number(file): 
    idx0 = file.index("exp")
    idx1 = file.index("_")
    return int(file[idx0+3:idx1])

def get_tracked_cliques(exp,sim_length,lambda_u,all_clique_feats,all_observations_dir,min_feats,max_feats,data_association): 
    all_observation_files = [x for x in os.listdir(all_observations_dir) if "observations" in x]
    '''
    #all_clique_feats[lm_id][feature id]["feat des"]/["feat_loc"]
	#obsd_cliques[exp][t] = observations
	#	observations[lm_id][feature_id]["feat_des"]
	#	observations[lm_id][feature_id]["feat_loc"]
    '''
    #all_observations are the observations of the current experiment 
    print("getting tracked cliques...")
    tracked_cliques = {}
    #clique_id, log_survival_function, num_feature, initialization_time
    for lm_id in all_clique_feats.keys():
        #print("this is exp: {} and this is lm_id: {}".format(exp,lm_id))
        if data_association is None:
            #print("calling get_reinitted_id with this data association: {}".format(data_association))
            c = get_reinitted_id(data_association,exp,lm_id)
        else:
            data_association_filepath = "/home/kristen/BetterFaster3.0/sim_utils/fake_data/data_associations/exp"+str(exp)+"_data_association.csv"
            data_association_dir = "/home/kristen/BetterFaster3.0/sim_utils/fake_data/data_associations" 
            #print("calling get_reinitted_id with: {}".format(np.genfromtxt(data_association_filepath)))      
            #print("np.genfromtxt(data_association_filepath): ",np.genfromtxt(data_association_filepath))
            all_data_associations = {} 
            for file in os.listdir(data_association_dir): 
                data_association_path = os.path.join(data_association_dir,file) 
                i0 = file.index("exp"); i1 = file.index("_data")
                exp_n = int(file[i0+3:i1])
                all_data_associations[exp_n] = np.genfromtxt(data_association_path)
            c = get_reinitted_id(all_data_associations,exp,lm_id)

        min_key = min(all_data_associations.keys())
        if not c in all_data_associations[min_key][:,0]:
            raise OSError 

        if max_feats < len(all_clique_feats[lm_id].keys()) and min_feats <= len(all_clique_feats[lm_id].keys()): 
            #randomly sample them 
            obsd_feat_ids = random.sample(range(0, max(all_clique_feats[lm_id].keys())), max_feats)
        else:
            obsd_feat_ids = list(all_clique_feats[lm_id].keys())
        
        
        potential_init_times = []
        #get the initialization time
        sorted_observation_files = sorted(all_observation_files, key=get_exp_number)
        #print("sorted_observation_files: ",sorted_observation_files)
        #print("max exp: ",max([get_exp_number(file) for file in all_observation_files]))
        
        for i in range(max([get_exp_number(file) for file in all_observation_files])): 
            '''
            if init_time is not None:
                print("init_time is not None!")
                print("init_time: ",init_time)
                break 
            '''
            with open(os.path.join(all_observations_dir,sorted_observation_files[i]),"rb") as handle:
                #print("checking {}".format(os.path.join(all_observations_dir,sorted_observation_files[i])))
                exp_observations = pickle.load(handle)
                t = 0
                init_time = None 
                while init_time is None:    
                    #print("all_observations:",all_observations)
                    #print("this is t: ",t)
                    if t in exp_observations.keys():
                        if isinstance(exp_observations[t],dict):
                            #see if it was observed in this timestep
                            if lm_id in exp_observations[t].keys():
                                #print("this is lm_id:",lm_id)
                                global_t = t + i*sim_length
                                init_time = global_t 
                                #print("init_time: ",init_time)
                                potential_init_times.append(global_t)
                                #print("appending to potential init times...")
                                break 
                        else:
                            if lm_id in [x['clique_id'] for x in exp_observations[t]]: 
                                #print("this is lm_id: ",lm_id)
                                global_t = t + i*sim_length
                                init_time = global_t
                                #print("init_time:",init_time)
                                potential_init_times.append(global_t)
                                #print("appending to potential init times...")
                                break 
                    else:
                        #print("we never observe {}".format(lm_id))
                        if lm_id == 0 and i == 0:
                            raise OSError
                        break 
                    t += 1

        #print("potential_init_times: ",potential_init_times)
        init_time = min(potential_init_times)

        if exp > 1 and lm_id == 0:
            if init_time != 0:
                raise OSError
        
        if not init_time is None:
            if c not in tracked_cliques.keys():
                tracked_cliques[c] = PersistenceFilter(lambda_u,num_features=max(all_clique_feats[lm_id].keys()),initialization_time=init_time)
            else:
                #im confused...  
                raise OSError 
    
    return tracked_cliques

def cone_gstate_function(x): 
    if x <= 9:
        return 1 
    else:
        return 0 

def tree_gstate_function(x): 
    if x <= 3:
        return 1
    elif 3 < x and x <= 6:
        return 2 
    elif 6 < x and x<= 9:
        return 3
    elif 9 < x:
        return 1  
