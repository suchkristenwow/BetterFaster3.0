from JointPersistenceFilter import PersistenceFilter
import random 
import numpy as np 
import sys
sys.path.append('../sim_utils')
from utils import get_reinitted_id
import os 


def get_tracked_cliques(exp,lambda_u,all_clique_feats,all_observations,min_feats,max_feats,data_association): 
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
            print("np.genfromtxt(data_association_filepath): ",np.genfromtxt(data_association_filepath))
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

        #print("this is lm_id:",lm_id)
        if len(all_clique_feats[lm_id].keys()) < min_feats:
            #we didnt really observe that many features for this... 
            continue 
        if max_feats < len(all_clique_feats[lm_id].keys()): 
            #randomly sample them 
            obsd_feat_ids = random.sample(range(0, max(all_clique_feats[lm_id].keys())), max_feats)
        else:
            obsd_feat_ids = list(all_clique_feats[lm_id].keys())
        
        #get the initialization time
        init_time = None 
        t = 0
        while init_time is None:    
            #print("all_observations:",all_observations)
            if t in all_observations.keys():
                if isinstance(all_observations[t],dict):
                    #see if it was observed in this timestep
                    if lm_id in all_observations[t].keys():
                        init_time = t 
                else:
                    if lm_id in [x['clique_id'] for x in all_observations[t]]: 
                        init_time = t 
            else:
                print("we never observe {}".format(lm_id))
                break 
            t += 1

        if not init_time is None:
            if c not in tracked_cliques.keys():
                tracked_cliques[c] = PersistenceFilter(lambda_u,num_features=max(all_clique_feats[lm_id].keys()),initialization_time=init_time)
            else:
                #im confused...  
                raise OSError 

    return tracked_cliques
