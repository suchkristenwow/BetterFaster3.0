from .JointPersistenceFilter import PersistenceFilter
import random 
import numpy as np 
import os 
import pickle 
from betterFaster.sim_utils.utils import get_reinitted_id 

def last_index_of_change(lst):
    for i in range(len(lst) - 1, 0, -1):
        if lst[i] != lst[i - 1]:
            return i
    return -1  # If there is no change

def get_exp_number(file): 
    #print("file: ",file)
    if "all_clique_feats" in file:  
        idx0 = file.index("experiment")
        idx1 = file.index("all_clique_feats")
        return int(file[idx0+10:idx1])
    else: 
        try:
            idx0 = file.index("exp")
            idx1 = file.index("observed")
            return int(file[idx0+3:idx1])
        except: 
            idx0 = file.index("experiment")
            idx1 = file.index("observed")
            return int(file[idx0+10:idx1])

def get_n_clique_feats(all_clique_feats,lm_id,max_feats,min_feats,data_associations=None):
    if data_associations is not None and lm_id not in all_clique_feats.keys(): 
        for i in data_associations.keys(): 
            reinitted_id = get_reinitted_id(data_associations,i,lm_id,optional_exp=i)
            if reinitted_id in all_clique_feats.keys(): 
                lm_id = reinitted_id 
                break 
    
    if lm_id not in all_clique_feats.keys():
        raise OSError 
    
    if max_feats < len(all_clique_feats[lm_id].keys()) and min_feats <= len(all_clique_feats[lm_id].keys()): 
        n_obsd_feat_ids = max_feats 
    else:
        #obsd_feat_ids = list(all_clique_feats[lm_id].keys())
        n_obsd_feat_ids = len([x for x in all_clique_feats[lm_id].keys()]) 
    return n_obsd_feat_ids 
    
def get_init_time(results_dir,n_experiments,exp,sim_length,lm_id,is_carla): 
    all_observations_dir = os.path.join(results_dir,"observation_pickles") 
    all_observation_files = [x for x in os.listdir(all_observations_dir)] 
    sorted_observation_files = sorted(all_observation_files, key=get_exp_number)
    potential_init_times = [] 

    for i in range(n_experiments):
        #print("this is experiment: ",i)
        with open(os.path.join(all_observations_dir,sorted_observation_files[i]),"rb") as handle:
            #print("checking {}".format(os.path.join(all_observations_dir,sorted_observation_files[i])))
            exp_observations = pickle.load(handle)
        if len(exp_observations.keys()) == 1 and is_carla:
            if int([x for x in exp_observations.keys()][0]) == exp:
                #print("exp_observations.keys(): ",exp_observations.keys())
                exp_observations = exp_observations[exp]
                #print("max([x for x in exp_observations.keys()]):",max([x for x in exp_observations.keys()]))
        t = 0
        init_time = None 
        while init_time is None and t < exp*sim_length + sim_length:    
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
            t += 1

        if len(potential_init_times) > 0:
            break 

    if len(potential_init_times) > 0:
        #print("potential_init_times: ",potential_init_times)
        init_time = min(potential_init_times)
    else:
        return None 
        
    return init_time 

def get_tracked_cliques(results_dir,n_experiments,exp,sim_length,parameters,all_data_associations,all_clique_feats,is_carla,exp_observed_cliques): 
    #print("getting tracked cliques: ",exp_observed_cliques)
    tracked_cliques = {}

    sim_length = parameters["sim_length"]
    lambda_u = parameters["betterTogether"]["lambda_u"]
    min_feats = parameters["betterTogether"]["min_feats"]
    max_feats = parameters["betterTogether"]["max_feats"] 
    all_observations_dir = os.path.join(parameters["results_dir"],"observation_pickles") 
    all_observation_files = [x for x in os.listdir(all_observations_dir) if "observed" in x]

    if is_carla: 
        if os.path.exists("tracked_clique_pickles/exp"+str(exp)+"tracked_cliques.pickle"): 
            with open("tracked_clique_pickles/exp"+str(exp)+"tracked_cliques.pickle","rb") as handle: 
                tracked_cliques_dict = pickle.load(handle)
            #tracked_cliques[c] = PersistenceFilter(lambda_u,num_features=n_obsd_feat_ids,initialization_time=init_time)
            for c in tracked_cliques_dict.keys(): 
                tracked_cliques[c] = PersistenceFilter(lambda_u,num_features=tracked_cliques_dict[c]["n_obsd_feats"],initialization_time=tracked_cliques_dict[c]["init_time"])
            return tracked_cliques 

    #all_observations are the observations of the current experiment 
    print("getting tracked cliques...")

    for x,lm_id in enumerate(all_clique_feats.keys()):
        #print("this is exp: {} and this is lm_id: {}".format(exp,lm_id))
        print("{} of {} landmarks ...".format(x,len(all_clique_feats.keys())))
        if all_data_associations is not None:
            #print("calling get_reinitted_id with this data association: {}".format(all_data_associations))
            c = get_reinitted_id(all_data_associations,exp,lm_id)
        else:
            data_associations_path = os.path.join(results_dir,"data_association/experiment"+str(exp)+"data_association.csv")
            if not os.path.exists(data_associations_path):
                data_associations_path = os.path.join(results_dir,"data_associations/exp"+str(exp-1)+"_data_association.csv")
                if not os.path.exists(data_associations_path):
                    print("data_associations_path:",data_associations_path)
                    raise OSError
            c = get_reinitted_id(all_data_associations,exp,lm_id)
        
        #print("this clique has {} features".format(len(all_clique_feats[lm_id].keys())))
        if max_feats < len(all_clique_feats[lm_id].keys()) and min_feats <= len(all_clique_feats[lm_id].keys()): 
            n_obsd_feat_ids = max_feats 
        else:
            #obsd_feat_ids = list(all_clique_feats[lm_id].keys())
            n_obsd_feat_ids = len([x for x in all_clique_feats[lm_id].keys()])

        potential_init_times = []
        #get the initialization time
        sorted_observation_files = sorted(all_observation_files, key=get_exp_number)

        for i in range(n_experiments):
            #print("this is experiment: ",i)
            #print("opening this pickle: ",os.path.join(all_observations_dir,sorted_observation_files[i]))
            with open(os.path.join(all_observations_dir,sorted_observation_files[i]),"rb") as handle:
                #print("checking {}".format(os.path.join(all_observations_dir,sorted_observation_files[i])))
                exp_observations = pickle.load(handle)

            #print("these are the keys: ",exp_observations.keys()) 

            if len(exp_observations.keys()) == 1 and is_carla:
                #print("there is only one key: {}, and this is is_carla: {}".format(exp_observations.keys(),is_carla)) 
                k = [x for x in exp_observations.keys()][0] 
                exp_observations = exp_observations[k]

            t = 0
            init_time = None 

            if is_carla: 
                tf = exp*sim_length + sim_length 
            else:
                tf = sim_length 

            while t < tf:     
                if init_time is not None:
                    break 

                if t in exp_observations.keys():
                    if isinstance(exp_observations[t],dict):
                        if lm_id in exp_observations[t].keys():
                            #print("this is lm_id:",lm_id)
                            global_t = t + i*sim_length
                            init_time = global_t 
                            #print("init_time: ",init_time)
                            potential_init_times.append(global_t)
                            #print("appending to potential init times... this is t: ",t)
                            break 
                        elif c in exp_observations[t].keys(): 
                            #print("this is lm_id:",c)
                            global_t = t + i*sim_length
                            init_time = global_t 
                            #print("init_time: ",init_time)
                            potential_init_times.append(global_t)
                            #print("appending to potential init times... this is t: ",t)
                            break 
                    else:
                        if lm_id in [x['clique_id'] for x in exp_observations[t]]: 
                            #print("this is lm_id: ",lm_id)
                            global_t = t + i*sim_length
                            init_time = global_t
                            #print("init_time:",init_time)
                            potential_init_times.append(global_t)
                            #print("appending to potential init times... this is t: ",t)
                            break
                        elif c in [x['clique_id'] for x in exp_observations[t]]: 
                            #print("this is lm_id: ",c)
                            global_t = t + i*sim_length
                            init_time = global_t
                            #print("init_time:",init_time)
                            potential_init_times.append(global_t)
                            #print("appending to potential init times... this is t: ",t)
                            break    
                elif lm_id in exp_observations.keys():
                    print("last_tstep: {}".format(exp*sim_length + sim_length))
                    print("min: {}, max: {}".format(min(exp_observations.keys()),max(exp_observations.keys())))
                    print("this is t: {}, this is sim_length: {}, and exp: {}".format(t,sim_length,exp))
                    print("lm_id: {}, exp_observations.keys(): ".format(lm_id,exp_observations.keys()))
                    raise OSError  
                else:
                    if len(exp_observations.keys()) != sim_length: 
                        print("lm_id: {}, exp_observations.keys(): ".format(lm_id,exp_observations.keys()))
                        raise OSError 

                t += 1 

        if len(potential_init_times) > 0:
            #print("potential_init_times: ",potential_init_times)
            init_time = min(potential_init_times)

        if exp > 1 and lm_id == 0:
            if init_time != 0:
                raise OSError

        if not init_time is None:
            if c not in tracked_cliques.keys():
                tracked_cliques[c] = PersistenceFilter(lambda_u,num_features=n_obsd_feat_ids,initialization_time=init_time)
            '''
            else:
                print("this is c: ",c)
                print("init_time: ",init_time)
                print("tracked_cliques.keys(): ",tracked_cliques.keys())
                print("tracked_cliques[c]: ",tracked_cliques[c])
                #im confused...  
                raise OSError 
            '''
        else:
            #print("this seems fucked up .................")
            for i in range(n_experiments):
                with open(os.path.join(all_observations_dir,sorted_observation_files[i]),"rb") as handle:
                    #print("checking {}".format(os.path.join(all_observations_dir,sorted_observation_files[i])))
                    exp_observations = pickle.load(handle)
                if len(exp_observations.keys()) == 1 and is_carla:
                    if int([x for x in exp_observations.keys()][0]) == exp:
                        #print("exp_observations.keys(): ",exp_observations.keys())
                        exp_observations = exp_observations[exp]
                #print("exp_observations.keys(): ",exp_observations.keys())
                for t in exp_observations.keys(): 
                    if lm_id or c in exp_observations.keys():
                        print("lm_id: {},c:{} observed at experiment: {}, timestep: {}".format(lm_id,c,i,t))
                        raise OSError 

            raise OSError 

    for c in exp_observed_cliques:  
        if c not in tracked_cliques.keys():
            for i in all_data_associations.keys():
                reinitted_id = get_reinitted_id(all_data_associations,exp,c)
                if reinitted_id in tracked_cliques.keys():
                    break 

            if reinitted_id not in tracked_cliques.keys():
                good_id = False 
                for i in all_data_associations:
                    reinitted_id = get_reinitted_id(all_data_associations,exp,c,optional_exp=i)
                    if reinitted_id in all_clique_feats:
                        good_id = True 
                        break 
                if good_id: 
                    n_obsd_feat_ids = get_n_clique_feats(all_clique_feats,reinitted_id,max_feats,min_feats,data_associations=all_data_associations)
                    init_time = get_init_time(results_dir,max(all_data_associations.keys()),exp,sim_length,reinitted_id,parameters["isCarla"]) 
                    tracked_cliques[c] = PersistenceFilter(lambda_u,num_features=n_obsd_feat_ids,initialization_time=init_time)
        
    if len(tracked_cliques.keys()) == 0:
        raise OSError 
    
    #debug 
    for c in tracked_cliques.keys():
        if len(tracked_cliques[c]._log_likelihood)  > max_feats:
            raise OSError 

    if is_carla: 
        #save this to go mo' faster
        if not os.path.exists("tracked_clique_pickles"):
            os.mkdir("tracked_clique_pickles")
        with open("tracked_clique_pickles/exp"+str(exp)+"tracked_cliques.pickle","wb") as handle: 
            #PersistenceFilter(lambda_u,num_features=n_obsd_feat_ids,initialization_time=init_time)
            tracked_cliques_dict = {}         
            for c in tracked_cliques.keys():
                tracked_cliques_dict[c] = {}
                tracked_cliques_dict[c]["n_obsd_feats"] = len(tracked_cliques[c]._log_likelihood) 
                tracked_cliques_dict[c]["init_time"] = tracked_cliques[c].init_tstep 
            pickle.dump(tracked_cliques_dict,handle)

    return tracked_cliques

def cone_gstate_function(tsteps_in_current_state): 
    #print("entered cone gstate estimate function ... this is tsteps in current state: ",tsteps_in_current_state)
    if tsteps_in_current_state <= 3:
        return 1 
    else:
        return 0 
    
def tree_gstate_function(x): 
    #print("this is tree_gstate_function ... experiment: ",x)
    if x <= 3:
        return 1
    elif 3 < x and x <= 6:
        return 2 
    elif 6 < x and x<= 9:
        return 3
    elif 9 < x:
        return 1  
