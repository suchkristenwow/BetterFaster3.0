from betterFaster import * 
import numpy as np 
from betterFaster.betterTogether.clique_utils import get_tracked_cliques, cone_gstate_function, tree_gstate_function, get_n_clique_feats, get_init_time  
from betterFaster.betterTogether.JointPersistenceFilter import PersistenceFilter
from betterFaster.sim_utils.utils import get_reinitted_id 
import os 
import threading 
import time  

class betterTogether_simulator: 
    def __init__(self,parameters,**kwargs): 
        self.parameters = parameters 
        self.P_Miss_detection = parameters["betterTogether"]["P_Miss_detection"]
        self.P_False_detection = parameters["betterTogether"]["P_False_detection"]
        self.acceptance_threshold = parameters["betterTogether"]["detection_threshold"]
        self.rejection_threshold = parameters["betterTogether"]["rejection_threshold"]
        self.min_feats = parameters["betterTogether"]["min_feats"]
        self.max_feats = parameters["betterTogether"]["max_feats"]
        self.confidence_range = parameters["betterTogether"]["confidence_range"]
        self.tune_bool = parameters["comparison_bools"]["compare_betterTogether"]
        self.clique_features = kwargs.get("all_clique_features")
        self.observed_ids = [x for x in self.clique_features.keys()]
        self.results_dir = parameters["results_dir"]
        self.current_exp = kwargs.get("current_exp")
        self.observations = kwargs.get("exp_observations")
        self.sim_length = kwargs.get("sim_length")
        self.verbose = kwargs.get("verbosity")
        self.enable_feature_detection = kwargs.get("feature_detection")
        self.lambda_u = parameters["betterTogether"]["lambda_u"]
        self.all_data_associations = kwargs.get("data_association")
        self.n_experiments = parameters["experiments"]
        self.isCarla = parameters["isCarla"]
        exp_obs_ids = kwargs.get("exp_obs_ids")
        start_times = kwargs.get("start_time")
        self.start_exp = start_times[0]; self.start_tstep = start_times[1]
        self.tracked_cliques = get_tracked_cliques(self.results_dir,self.n_experiments,self.current_exp,self.sim_length,parameters,self.all_data_associations,self.clique_features,self.isCarla,exp_obs_ids)
        self.posteriors = {}
        self.growth_state_estimates = {} 
        for c in self.tracked_cliques.keys():
            self.posteriors[c] = np.zeros((self.sim_length*self.n_experiments,))
            self.growth_state_estimates[c] = np.zeros((self.sim_length*self.n_experiments,))
        #GROWTH STATE TRACKING STUFF 
        #parameters["betterFaster"]["T_nu_"] is the mean number of experiments the landmarks persist 
        self.T_nu_cone = parameters["betterFaster"]["T_nu_cone"]*self.sim_length 
        self.T_nu_tree = parameters["betterFaster"]["T_nu_tree"]*self.sim_length 
        self.delta_t_gstate_samples = {} 
        self.delta_t_gstate_samples["cone"] = []
        self.delta_t_gstate_samples["tree"] = []
        self.cone_feature_description_cache = {} 
        self.tree_feature_description_cache = {} 
        self.growth_state_relevancy_level = parameters["betterFaster"]["growth_state_relevancy_level"]
        self.n_gstates = parameters["betterFaster"]["n_gstates"]
        self.hamming_similarity_val = parameters["betterFaster"]["hamming_distance_similarity"]
        self.exceeded_Tnu_cone_tsteps = 0 
        self.exceeded_Tnu_tree_tsteps = 0

        #extract ground truth landmarks 
        #print("finding cone ids for {}".format(self.current_exp + 1))
        cone_ids_file = os.path.join(self.results_dir,"cone_ids/experiment"+str(self.current_exp + 1)+"cone_ids_.txt")
        # Read the contents of the text file
        with open(cone_ids_file, 'r') as file:
            lines = file.readlines()
        self.cone_ids = np.unique([int(line.strip()) for line in lines])

        #print("finding tree ids for {}".format(self.current_exp + 1))
        # Convert each line to an integer and store in a list
        tree_ids_file = os.path.join(self.results_dir,"tree_ids/experiment"+str(self.current_exp + 1)+"tree_ids_.txt")
        # Read the contents of the text file
        with open(tree_ids_file, 'r') as file:
            lines = file.readlines()
        # Convert each line to an integer and store in a list
        self.tree_ids = np.unique([int(line.strip()) for line in lines]) 

        self.ever_obsd = []

        self.observation_cache = {} 

        #Trying to go fast:
        self.get_detection_lists_times = []
        self.gstate_estimate_times = []
        self.prediction_times = []
        self.update_times = []
        self.check_times = []

        self.new_exp_ids = [] 
        
    #exp,exp_observations,all_clique_feats,observed_clique_id
    def reinit_experiment(self,exp,observations,all_clique_feats,observed_clique_ids): 
        print("reinitting experiment...")
        self.current_exp = exp 
        self.observations = observations 

        new_tracked_cliques = get_tracked_cliques(self.results_dir,self.n_experiments,exp,self.sim_length,self.parameters,self.all_data_associations,self.clique_features,self.isCarla,observed_clique_ids)
        #print("new_tracked_cliques.keys(): ",new_tracked_cliques.keys())

        #debugging
        for c in new_tracked_cliques.keys():
            if np.any(new_tracked_cliques[c]._last_observation_time) is None:
                print("new_tracked_cliques[c]._last_observation_time: ",new_tracked_cliques[c]._last_observation_time)
                raise OSError  
            '''
            else:
                print("new_tracked_cliques[c]._last_observation_time[0]: ",new_tracked_cliques[c]._last_observation_time[0])
            '''

        #print("previous clique feature keys: ",self.clique_features.keys())
        self.clique_features = all_clique_feats 

        for c in new_tracked_cliques.keys():
            reinitted_id = get_reinitted_id(self.all_data_associations,exp,c,optional_exp=exp) 
            orig_id = get_reinitted_id(self.all_data_associations,exp,c,optional_exp=0)
            valid_id = True
            if c not in self.tracked_cliques.keys():
                if reinitted_id not in self.tracked_cliques.keys(): 
                    #this is a new clique 
                    if c in self.clique_features.keys():
                        n_obsd_feat_ids = get_n_clique_feats(self.clique_features,c,self.max_feats,self.min_feats)
                        init_time = get_init_time(self.results_dir,self.n_experiments,self.current_exp,self.sim_length,c,self.isCarla)
                    else: 
                        if reinitted_id in self.clique_features.keys(): 
                            n_obsd_feat_ids = get_n_clique_feats(self.clique_features,reinitted_id,self.max_feats,self.min_feats)  
                            init_time = get_init_time(self.results_dir,self.n_experiments,self.current_exp,self.sim_length,reinitted_id,self.isCarla) 
                        elif orig_id in self.clique_features.keys(): 
                            n_obsd_feat_ids = get_n_clique_feats(self.clique_features,orig_id,self.max_feats,self.min_feats) 
                            init_time = get_init_time(self.results_dir,self.n_experiments,self.current_exp,self.sim_length,orig_id,self.isCarla) 
                        else:
                            valid_id = False
                            for i in self.all_data_associations.keys(): 
                                id_ = get_reinitted_id(self.all_data_associations,exp,c,optional_exp=i)
                                if id_ in self.clique_features.keys():
                                    #print("id_ is in self.clique_features!")
                                    valid_id = True 
                                    break  
                            if valid_id: 
                                n_obsd_feat_ids = get_n_clique_feats(self.clique_features,c,self.max_feats,self.min_feats,data_associations=self.all_data_associations)
                                init_time = get_init_time(self.results_dir,self.n_experiments,self.current_exp,self.sim_length,c,self.isCarla) 
                    if valid_id: 
                        self.tracked_cliques[c] = PersistenceFilter(self.lambda_u,num_features=n_obsd_feat_ids,initialization_time=init_time) 
                else: 
                    #so c is in new_tracked_cliques.keys and reinitted_id is in self.tracked_cliques.keys() 
                    old_clique_likelihood = self.tracked_cliques[reinitted_id]._clique_likelihood 
                    old_log_clique_evidence = self.tracked_cliques[reinitted_id]._log_clique_evidence 
                    old_last_observation_time = self.tracked_cliques[reinitted_id]._last_observation_time 
                    if old_last_observation_time is None:
                        raise OSError 
                    old_log_likelihood = self.tracked_cliques[reinitted_id]._log_likelihood 
                    old_log_clique_lower_evidence_sum = self.tracked_cliques[reinitted_id]._log_clique_lower_evidence_sum 
                    #print("reinitted id is in self.tracked_cliques.keys(), updating it".format(reinitted_id))
                    self.tracked_cliques[reinitted_id] = new_tracked_cliques[c]
                    self.tracked_cliques[reinitted_id]._clique_likelihood = old_clique_likelihood 
                    self.tracked_cliques[reinitted_id]._log_clique_evidence = old_log_clique_evidence 
                    self.tracked_cliques[reinitted_id]._last_observation_time = old_last_observation_time
                    self.tracked_cliques[reinitted_id]._log_likelihood = old_log_likelihood
                    self.tracked_cliques[reinitted_id]._log_clique_lower_evidence_sum = old_log_clique_lower_evidence_sum  
            else: 
                #load in all the old stuff from the previous experiment 
                old_clique_likelihood = self.tracked_cliques[c]._clique_likelihood 
                old_log_clique_evidence = self.tracked_cliques[c]._log_clique_evidence 
                old_last_observation_time = self.tracked_cliques[c]._last_observation_time 
                if old_last_observation_time is None:
                    raise OSError 
                old_log_likelihood = self.tracked_cliques[c]._log_likelihood 
                old_log_clique_lower_evidence_sum = self.tracked_cliques[c]._log_clique_lower_evidence_sum 
                #print("c is not in self.tracked_cliques.keys(): ".format(c))
                self.tracked_cliques[c] = new_tracked_cliques[c]
                self.tracked_cliques[c]._clique_likelihood = old_clique_likelihood 
                self.tracked_cliques[c]._log_clique_evidence = old_log_clique_evidence 
                self.tracked_cliques[c]._last_observation_time = old_last_observation_time
                self.tracked_cliques[c]._log_likelihood = old_log_likelihood
                self.tracked_cliques[c]._log_clique_lower_evidence_sum = old_log_clique_lower_evidence_sum  
                
            if c not in self.posteriors.keys(): 
                #get_reinitted_id(self.all_data_associations,self.current_exp,c)
                reinitted_id = get_reinitted_id(self.all_data_associations,exp,c)
                if reinitted_id not in self.posteriors.keys(): 
                    self.posteriors[c] = np.zeros((self.sim_length*self.n_experiments,))
            if c not in self.growth_state_estimates.keys(): 
                reinitted_id = get_reinitted_id(self.all_data_associations,exp,c)
                if reinitted_id not in self.growth_state_estimates.keys(): 
                    self.growth_state_estimates[c] = np.zeros((self.sim_length*self.n_experiments,)) 

        purge_ids = []
        for c in self.tracked_cliques.keys(): 
            orig_id = get_reinitted_id(self.all_data_associations,exp,c) 
            reinitted_id =  get_reinitted_id(self.all_data_associations,exp,c,optional_exp=self.current_exp)
            if np.any(self.tracked_cliques[c]._last_observation_time) is None:
                if c in self.observation_cache.keys(): 
                    if len(self.observation_cache[c]) == 0:
                        raise OSError   
                    last_obsd_time = max(self.observation_cache[c])
                    self.tracked_cliques[c]._last_observation_time = [last_obsd_time for _ in self.tracked_cliques[c]._last_observation_time]
                else:
                    if reinitted_id in self.observation_cache.keys(): 
                        last_obsd_time = max(self.observation_cache[reinitted_id]) 
                        self.tracked_cliques[c]._last_observation_time = [last_obsd_time for _ in self.tracked_cliques[c]._last_observation_time]  
                    elif orig_id in self.observation_cache.keys(): 
                        last_obsd_time = max(self.observation_cache[reinitted_id]) 
                        self.tracked_cliques[c]._last_observation_time = [last_obsd_time for _ in self.tracked_cliques[c]._last_observation_time]  
                    else: 
                        found_init_time = False 
                        for x in self.all_data_associations.keys():
                            reinitted_id = get_reinitted_id(self.all_data_associations,exp,c,optional_exp=x) 
                            if reinitted_id in self.observation_cache.keys(): 
                                last_obsd_time = max(self.observation_cache[reinitted_id]) 
                                self.tracked_cliques[c]._last_observation_time = [last_obsd_time for _ in self.tracked_cliques[c]._last_observation_time]  
                                found_init_time = True 
                                break 
                        
                        if not found_init_time: 
                            init_time = get_init_time(self.results_dir,self.n_experiments,self.current_exp,self.sim_length,c,self.isCarla)
                            if init_time is not None: 
                                '''
                                print("WARNING: removing c:{} from tracked cliques".format(c))
                                    #del self.tracked_cliques[c]
                                    if c not in purge_ids:
                                        purge_ids.append(c)
                                else: 
                                '''
                                global_t = exp*self.sim_length
                                if init_time < global_t:
                                    print("init_time: {},global_t: {}".format(init_time,global_t))
                                    print("self.observation_cache.keys():",self.observation_cache.keys())
                                    print("self.tracked_cliques.keys(): ",self.tracked_cliques.keys()) 
                                    print("self.tracked_cliques[c].init_tstep: ",self.tracked_cliques[c].init_tstep)
                                    print("c: {}, orig_id: {}, reinitted_id: {}".format(c,orig_id,reinitted_id))
                                    raise OSError 
                                self.tracked_cliques[c]._last_observation_time = [init_time for _ in self.tracked_cliques[c]._last_observation_time]
        
        self.new_exp_ids = []
        for c in observed_clique_ids: 
            if c not in self.tracked_cliques.keys():
                n_obsd_feat_ids = get_n_clique_feats(self.clique_features,c,self.max_feats,self.min_feats)
                self.tracked_cliques[c] = PersistenceFilter(self.lambda_u,num_features=n_obsd_feat_ids,initialization_time=exp*self.sim_length)  
                if c not in self.posteriors.keys(): 
                    self.posteriors[c] = np.zeros((self.sim_length*self.n_experiments,))
                if c not in self.growth_state_estimates.keys(): 
                    self.growth_state_estimates[c] = np.zeros((self.sim_length*self.n_experiments,))
                self.new_exp_ids.append(c)
                '''
                print("c: ",c)
                reinitted_id = get_reinitted_id(self.all_data_associations,self.current_exp,c,optional_exp=self.current_exp)
                orig_id = get_reinitted_id(self.all_data_associations,self.current_exp,c)
                print("orig_id: {},reinitted_id: {}".format(orig_id,reinitted_id))
                print("self.tracked_cliques.keys(): ",self.tracked_cliques.keys())
                raise OSError  
                '''

    def normalize(self,global_t):
        if self.verbose:
            print("normalizing ... ")
            print("[self.tracked_cliques[x].init_tstep for x in self.tracked_cliques.keys()]:",[self.tracked_cliques[x].init_tstep for x in self.tracked_cliques.keys()]) 
            print("global_t: ",global_t)

        observed_clique_id = [x for x in self.tracked_cliques.keys() if self.tracked_cliques[x].init_tstep <= global_t] 
        #print("these should be the cliques we already observed: ",observed_clique_id)
    
        observed_posteriors = [self.posteriors[x][global_t] for x in observed_clique_id]
        #print("len(observed_posteriors): ",len(observed_posteriors))
        #print("observed_posteriors: ",observed_posteriors)

        if len(observed_posteriors) > 0:
            norm_factor = max(observed_posteriors)

            normalized_posteriors = []
            if self.verbose: 
                print("iterating through {} posteriors".format(len(self.posteriors.keys()))) 
            for c in self.posteriors.keys(): 
                if c not in observed_clique_id:
                    #print("we havent observed this yet") 
                    normalized_posteriors.append(1.0)
                else:
                    p_c = self.posteriors[c][global_t]
                    if norm_factor == 0:
                        print("WARNING NORMALIZATION FACTOR IS 0?")
                        #input("Press Enter to Continue ...")
                        p_c = 1 
                    else: 
                        p_c = p_c / norm_factor 
                    if np.isnan(p_c):
                        raise OSError 
                    elif np.isinf(p_c): 
                        raise OSError
                    #print("this is normalized: ",p_c)
                    normalized_posteriors.append(p_c)

        else: 
            #print("literally nothing has ever been observed ever")
            normalized_posteriors = [self.posteriors[x][global_t] if self.posteriors[x][global_t] <= 1 else 1 for x in self.tracked_cliques.keys()]

        if len(normalized_posteriors) != len(self.posteriors.keys()): 
            raise OSError 
        
        return normalized_posteriors 
    
    def get_detection_lists(self,detections_t,observed_cliques): 
        detection_lists = {} 
        for c in observed_cliques:
            orig_id = c
            if c not in self.clique_features.keys():
                #print("get_reinitted_id... this is c:",c) 
                c = get_reinitted_id(self.all_data_associations,self.current_exp,c)
                if c is None:
                    raise OSError 
                if c not in self.clique_features.keys():
                    #print("this is reinitted_c: ",c)
                    #print("self.clique_features.keys():",self.clique_features.keys())    
                    for i in self.all_data_associations.keys(): 
                        #print("trying to get reinitted id ... this is exp: {}".format(i))
                        c = get_reinitted_id(self.all_data_associations,self.current_exp,orig_id,optional_exp=i)
                        #print("this is reinitted id: ",c)
                        if c in self.clique_features: 
                            #print("found correct reinitted id!")
                            break 
            
            if c not in self.clique_features.keys():
                print("observed_cliques: ",observed_cliques)
                print("self.clique_features.keys(): ",self.clique_features.keys())
                raise OSError 

            if self.enable_feature_detection:
                detection_lists[c] = [0 for _ in self.clique_features[c].keys()]
            else:
                detection_lists[c] = [1 for _ in self.clique_features[c].keys()]
        
        if self.enable_feature_detection: 
            for el in detections_t:
                if el['detection']:
                    if el['clique_id'] in detection_lists.keys():
                        if el['feature_id']-1 < len(detection_lists[el['clique_id']]): #or np.random.rand() < self.P_Miss_detection:
                            detection_lists[el['clique_id']][el['feature_id']-1] = 1 

        return detection_lists

    def remove_bad_single_feat_detections(self, detection_lists, observed_cliques, detections_t):
        # Remove bad single feature detections
        bad_single_feat_detections = []
        not_obsd_cliques = []
        # Iterate over observed_cliques and detection_lists simultaneously using zip
        for k, (clique_id, el) in enumerate(zip(observed_cliques, detections_t)):
            # Get detection list corresponding to clique_id
            detection_list = detection_lists.get(clique_id)
            if detection_list is not None:
                if np.sum(detection_list) <= 1 and self.confidence_range < el['range']:
                    bad_single_feat_detections.append(k)
                    not_obsd_cliques.append(clique_id)

        # Use dictionary comprehension to filter detection_lists
        detection_lists = {key: value for idx, (key, value) in enumerate(detection_lists.items()) if idx not in bad_single_feat_detections}
        # Filter observed_cliques list
        observed_cliques = [x for idx, x in enumerate(observed_cliques) if idx not in bad_single_feat_detections]

        return detection_lists, observed_cliques

    def parallel_update_helper(self,c,detection_list,t): 
        self.tracked_cliques[c].update(detection_list,t,self.P_Miss_detection,self.P_False_detection)

    def update(self,t,detections_t):
        if self.verbose and np.mod(t,10) == 0: 
            if len(self.get_detection_lists_times) > 10:
                print("Detection times: ",np.mean(self.get_detection_lists_times))
                print("Update times: ",np.mean(self.gstate_estimate_times))
                print("Prediction times: ",np.mean(self.prediction_times))
                print("Gstate estimation times: ",np.mean(self.gstate_estimate_times))
                print("Debug check times: ",np.mean(self.check_times))

        t0 = time.time()

        observed_cliques = np.unique([x['clique_id'] for x in detections_t]) 
        
        if self.verbose: 
            print("observed these cliques this timestep: ",observed_cliques)
        
        self.ever_obsd.extend([x for x in observed_cliques if x not in self.ever_obsd])

        #Get detection lists 
        detection_lists = self.get_detection_lists(detections_t,observed_cliques)

        #Remove bad single feat detections
        detection_lists,observed_cliques = self.remove_bad_single_feat_detections(detection_lists,observed_cliques,detections_t)
        self.get_detection_lists_times.append(time.time() - t0)

        global_t = self.current_exp*self.sim_length + t 

        if global_t < 0:
            raise OSError
        
        for c in observed_cliques:
            if c not in self.observation_cache.keys(): 
                self.observation_cache[c] = [global_t]
            else:
                self.observation_cache[c].append(global_t)

        #Update observed cliques 
        t0 = time.time() 
        threads = []
        
        for c in observed_cliques:
            reinitted_c = None 
            if c not in detection_lists.keys(): 
                orig_c = c 
                #print("detection_lists.keys():",detection_lists.keys())
                #print("this is c: ",c)
                for i in self.all_data_associations.keys(): 
                    #print("trying to find reinitted id ... this is exp: {}".format(i))
                    reinitted_c = get_reinitted_id(self.all_data_associations,self.current_exp,orig_c,optional_exp=i) 
                    #print("reinitted_id: ",c)
                    if reinitted_c in detection_lists.keys():
                        #print("breaking ...")
                        break 

            if reinitted_c is not None: 
                detection_lists[c] = detection_lists[reinitted_c]

            if c not in self.tracked_cliques.keys():
                orig_c = c 
                #print("this is c: ",c)
                #print("self.tracked_cliques.keys(): ",self.tracked_cliques.keys())
                valid_id = False 
                reinitted_c = get_reinitted_id(self.all_data_associations,self.current_exp,orig_c,optional_exp=self.current_exp)
                orig_c = get_reinitted_id(self.all_data_associations,self.current_exp,orig_c)  
                if reinitted_c not in self.tracked_cliques.keys() and orig_c not in self.tracked_cliques.keys():  
                    for i in self.all_data_associations.keys(): 
                        reinitted_c = get_reinitted_id(self.all_data_associations,self.current_exp,orig_c,optional_exp=i) 
                        if reinitted_c in self.tracked_cliques.keys():
                            valid_id = True 
                            break

                if not valid_id: 
                    print("id is invalid!")
                    n_obsd_feat_ids = get_n_clique_feats(self.clique_features,c,self.max_feats,self.min_feats) 
                    init_time = get_init_time(self.results_dir,self.n_experiments,self.current_exp,self.sim_length,c,self.isCarla) 
                    self.tracked_cliques[orig_c] = PersistenceFilter(self.lambda_u,num_features=n_obsd_feat_ids,initialization_time=init_time)  
                    c = orig_c; 

                    if orig_c not in self.posteriors.keys():
                        self.posteriors[orig_c] = np.zeros((self.sim_length*self.n_experiments,))
                        self.growth_state_estimates[orig_c] = np.zeros((self.sim_length*self.n_experiments,))  
                else: 
                    if reinitted_c in self.tracked_cliques.keys(): 
                        c = reinitted_c     

            if c not in self.tracked_cliques.keys():
                raise OSError 
            
            thread_c = None 
            if c not in detection_lists.keys():
                #print("This is c: ",c)
                #print("detection_lists.keys():",detection_lists.keys())
                for i in self.all_data_associations.keys(): 
                    reinitted_c = get_reinitted_id(self.all_data_associations,self.current_exp,orig_c,optional_exp=i) 
                    #print("reinitted_id: ",reinitted_c)
                    if reinitted_c in detection_lists.keys():
                        thread_c = threading.Thread(target=self.parallel_update_helper,args=(c,detection_lists[reinitted_c],global_t))
                        #print("breaking ...")
                        break 
            else: 
                thread_c = threading.Thread(target=self.parallel_update_helper, args=(c,detection_lists[c],global_t))

            if thread_c is not None: 
                #print("adding thread!")
                threads.append(thread_c)
                thread_c.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        self.update_times.append(time.time() - t0)

        t0 = time.time() 
        #predict clique likelihoods
        for c in self.tracked_cliques.keys():
            if c not in self.posteriors.keys():
                #print("adding new posterior key: ",c)
                self.posteriors[c] = np.zeros((self.sim_length*self.n_experiments,))
            posterior_c = self.tracked_cliques[c].predict_clique_likelihood(global_t)
            if np.isnan(posterior_c):
                print("clique likelihood is nan: ",)
                raise OSError 
            self.posteriors[c][global_t] = posterior_c
            #print("the posterior of clique {} is {}".format(c,self.posteriors[c][global_t]))
        self.prediction_times.append(time.time() - t0)

        #normalize posteriors 
        posteriors_t = [self.posteriors[c][global_t] for c in self.posteriors.keys()] #len (self.posteriors.keys())
        
        t0 = time.time()
        if len([x for x in posteriors_t if x > 1]):
            #print("this is posteriors_t before normalization:",posteriors_t)
            normalized_posteriors = self.normalize(global_t)
            if len(normalized_posteriors) != len(self.posteriors.keys()): 
                raise OSError 
            if np.isnan(normalized_posteriors).any():
                raise OSError 
            posteriors_t = normalized_posteriors 
        
        if self.verbose:
            print("There are {} posteriors".format(len(self.posteriors.keys())))
        
        reinit_ids = []
        for i,c in enumerate(self.posteriors.keys()):
            #print("len(posteriors_t):",len(posteriors_t))
            self.posteriors[c][global_t] = posteriors_t[i]
            if c in observed_cliques and self.posteriors[c][global_t] < 0.1:  
                print("this is c:{} and the posterior: {}".format(c,self.posteriors[c][global_t]))
                print("WHY IS THIS POSTERIOR SO LOW LOW EVEN THOUGH THE CLIQUE IS BEING OBSERVED?")
                reinit_ids.append(c) 

        persistent_obs = []
        non_persistent_cliques = []
        for x in detections_t: 
            if x['clique_id'] in self.posteriors.keys():
                if self.posteriors[x['clique_id']][global_t] > self.acceptance_threshold: 
                    persistent_obs.append(x) 
                if self.posteriors[x['clique_id']][global_t] < self.rejection_threshold: 
                    if x['clique_id'] not in non_persistent_cliques:
                        non_persistent_cliques.append(x['clique_id'])
            else:
                #get_reinitted_id(self.all_data_associations,self.current_exp,x['clique_id']) == c])
                for i in self.all_data_associations.keys(): 
                    reinitted_id = get_reinitted_id(self.all_data_associations,self.current_exp,x['clique_id'],optional_exp=i)  
                    if reinitted_id in self.posteriors.keys(): 
                        if self.posteriors[reinitted_id][global_t] > self.acceptance_threshold: 
                            persistent_obs.append(x) 
                        if self.posteriors[reinitted_id][global_t] < self.rejection_threshold:
                            if reinitted_id not in non_persistent_cliques:
                                non_persistent_cliques.append(reinitted_id) 
                        break 
                    '''
                    else:
                        print("reinitted_id: ",reinitted_id)
                        print("x: ",x)
                        print("self.posteriors.keys(): ",self.posteriors.keys())
                        raise OSError 
                    '''

        #print("these are the non persistent cliques:",non_persistent_cliques)
        for c in non_persistent_cliques: 
            if c not in self.growth_state_estimates.keys(): 
                reinitted_id = get_reinitted_id(self.all_data_associations,self.current_exp,c)
                if reinitted_id not in self.growth_state_estimates.keys():
                    raise OSError 
            #print("clique c: {} posterior is below the rejection threshold!".format(c))
            self.growth_state_estimates[c][global_t] = 0

        #print("this is non persistent cliques: ",non_persistent_cliques)
        current_gstate_estimates = []
        for c in self.posteriors.keys():
            if c in non_persistent_cliques:
                #print("c: {} is in non persistent cliques".format(c))
                continue 
            if c not in self.growth_state_estimates.keys():
                for i in self.all_data_associations.keys():
                    reinit_id = get_reinitted_id(self.all_data_associations,self.current_exp,c,optional_exp=i) 
                    if reinit_id in self.growth_state_estimates.keys():
                        c = reinit_id 
                        break 
                if c not in self.growth_state_estimates.keys():
                    self.growth_state_estimates[c] = np.zeros((self.sim_length*self.n_experiments,))
                    self.new_exp_ids.append(c)

            #print("self.start_exp: {},self.current_exp: {}, self.start_tstep: {}".format(self.start_exp,self.current_exp,self.start_tstep))
            if c in self.new_exp_ids: 
                estimated_gstate = self.sample_gstate(c)
            else:
                if self.start_exp is not None: 
                    if self.start_exp == self.current_exp and self.start_tstep == t: 
                        #print("sampling gstate...")
                        estimated_gstate = self.sample_gstate(c)  
                    else: 
                        #print("estimating growth state...")
                        estimated_gstate = self.estimate_growth_state(t,c,detections_t,self.posteriors[c][global_t]) 
                else:
                    #print("estimating growth state...")
                    estimated_gstate = self.estimate_growth_state(t,c,detections_t,self.posteriors[c][global_t])

            if self.posteriors[c][global_t] > self.rejection_threshold and estimated_gstate == 0:
                estimated_gstate = self.sample_gstate(c)
                
            current_gstate_estimates.append(estimated_gstate)

            if estimated_gstate > 1 and global_t < 10:
                print("growth state estimation is out of control")
                raise OSError 
            
            self.growth_state_estimates[c][global_t] = estimated_gstate 

        #print("current_gstate_estimates: ",current_gstate_estimates)
        if np.all(current_gstate_estimates == 0): 
            print("current_gstate_estimates: ",current_gstate_estimates)
            print("something is wrong! all landmarks are dead?")
            raise OSError 
        
        self.check_times.append(time.time() - t0)

        t0 = time.time()        
        if not self.tune_bool: 
            persistent_obs = []
            for c in self.growth_state_estimates.keys(): 
                if self.growth_state_estimates[c][global_t] != 0: 
                    persistent_obs.extend([x for x in detections_t if get_reinitted_id(self.all_data_associations,self.current_exp,x['clique_id']) == c])
        self.gstate_estimate_times.append(time.time() - t0)

        return persistent_obs, reinit_ids 

    def last_index_of_change(self,id_,t,prev_gstates_id):
        global_t = self.current_exp * self.sim_length + t 

        #print("Trying to find out how long this clique has been in this clique state")
        if t > 0:
            if id_ not in self.growth_state_estimates.keys(): 
                for i in self.all_data_associations.keys():
                    reinit_id = get_reinitted_id(self.all_data_associations,self.current_exp,id_,optional_exp=i)
                    if reinit_id in self.growth_state_estimates.keys():
                        id_ = reinit_id 
                        break 
            
            if id_ not in self.growth_state_estimates.keys():
                self.growth_state_estimates[id_] = np.zeros((self.sim_length*self.n_experiments,)) 
                return global_t 
            
            current_gstate = self.growth_state_estimates[id_][t-1]
        else:
            current_gstate = 1 
        #print("this is current_gstate: ",current_gstate)

        if not isinstance(self.all_data_associations,dict): 
            print("self.all_data_associations: ",self.all_data_associations)
            raise OSError 
        
        if not id_ in self.cone_ids and id_ not in self.tree_ids: 
            orig_id = id_ 
            for i in self.all_data_associations.keys(): 
                id_ = get_reinitted_id(self.all_data_associations,self.current_exp,orig_id,optional_exp=i) 
                if id_ in self.cone_ids or id_ in self.tree_ids: 
                    break 

        if id_ in self.cone_ids:
            if current_gstate == 1:
                prev_gstate = 0 
            else:
                prev_gstate = 1
        elif id_ in self.tree_ids:
            if current_gstate == 1: 
                if not self.n_gstates in prev_gstates_id: 
                    return 0 
            current_idx = np.where(np.arange(self.n_gstates) == current_gstate)
            prev_gstate = np.arange(self.n_gstates)[int(current_idx[0]) - 1] + 1 
        else:
            current_gstate = 0
            prev_gstate = None 
            for i in range(len(prev_gstates_id) - 1, -1, -1):
                if prev_gstates_id[i] != 0:
                    prev_gstate = prev_gstates_id[i]
                    break 
            if prev_gstate is None: 
                prev_gstate = 1 #idk anymore 

        #print("this is prev_gstate: ",prev_gstate)
        if prev_gstate in prev_gstates_id: 
            for i in range(len(prev_gstates_id) - 1, -1, -1):
                #print("prev_gstates_id: ",prev_gstates_id[i]) 
                #print("prev_gstate:",prev_gstate)
                if prev_gstates_id[i] == prev_gstate:
                    if i > global_t:
                        raise OSError 
                    return i
        else: 
            if 0 < self.current_exp*self.sim_length:
                return self.current_exp*self.sim_length
            else:
                return 0  

    def estimate_growth_state(self,t,id_,detections_t,posterior): 
        #print("estimating growth state of clique {}....".format(id_))
        #need to get time in the current growth state, then compare to T_nu_lmType  

        global_t = self.current_exp*self.sim_length + t 

        if not id_ in self.tree_ids and id_ not in self.cone_ids: 
            orig_id = id_ 
            id_ = get_reinitted_id(self.all_data_associations,self.current_exp,id_,self.current_exp) 
            #print("id_ was not in either cone or tree ids... this is reinitted id: ",id_)

        #print("this is global_t: ",global_t)

        if id_ in self.cone_ids:
            T_nu = self.T_nu_cone  
        elif id_ in self.tree_ids: 
            T_nu = self.T_nu_tree  
            
        if global_t == 0:
            prev_gstates_id = []
            d_t = 0 
        else:
            if id_ in self.growth_state_estimates.keys():
                prev_gstates_id = self.growth_state_estimates[id_][:global_t]
                if self.last_index_of_change(id_,t,prev_gstates_id) > global_t:
                    print("self.last_index_of_change: {}, global_t: {}".format(self.last_index_of_change(id_,t,prev_gstates_id),global_t))
                    print("this isnt possible")
                    raise OSError 
                #print("self.current_exp*self.sim_length: {}, self.last_index_of_change: {}".format((self.current_exp)*self.sim_length, self.last_index_of_change(id_,t,prev_gstates_id)))
                d_t = global_t - self.last_index_of_change(id_,t,prev_gstates_id)
            elif orig_id in self.growth_state_estimates.keys():  
                prev_gstates_id = self.growth_state_estimates[orig_id][:global_t]
                if self.last_index_of_change(orig_id,t,prev_gstates_id) > global_t:
                    print("self.last_index_of_change: {}, global_t: {}".format(self.last_index_of_change(orig_id,t,prev_gstates_id),global_t))
                    print("this isnt possible")
                    raise OSError 
                #print("self.current_exp*self.sim_length: {}, self.last_index_of_change: {}".format((self.current_exp)*self.sim_length, self.last_index_of_change(id_,t,prev_gstates_id)))
                d_t = global_t - self.last_index_of_change(id_,t,prev_gstates_id)
            else:
                print("orig_id: {}, id_:{}".format(orig_id,id_))
                print("self.growth_state_estimates.keys():",self.growth_state_estimates.keys())
                raise OSError 

        #print("this is d_t: ",d_t)
        if d_t < 0 or d_t > global_t: 
            print("this is d_t: {}, and this is global_t: {}".format(d_t,global_t)) 
            raise OSError 
        
        if global_t > 0:
            if not id_ in self.growth_state_estimates.keys():
                current_gstate = self.growth_state_estimates[orig_id][global_t-1]
            else: 
                current_gstate = self.growth_state_estimates[id_][global_t-1]
        else:
            current_gstate = 1 
        #print("current_gstate: ",current_gstate)

        if id_ in self.tree_ids:
            if current_gstate == 0 and posterior > self.acceptance_threshold:
                #print("woops we thought this was dead... we probably missed up bc the posterior is really high") 
                #print("this is gstate: ",self.sample_gstate(id_))
                return self.sample_gstate(id_) 
            elif current_gstate == 0 and posterior < self.acceptance_threshold: 
                #print("nah this is prolly dead fr")
                return 0 
            elif 0 < current_gstate:
                #print("in this statement! 0 < current_gstate")
                if current_gstate + 1 <= 3:
                    #print("current_gstate+1: ",current_gstate + 1)
                    next_gstate = current_gstate + 1
                else:
                    #print("next_gstate is one")
                    next_gstate = 1
            elif current_gstate < 0:
                raise OSError 
        elif id_ in self.cone_ids:
            if current_gstate == 0:
                next_gstate = 1 
            elif current_gstate < 0:
                raise OSError 
            else:
                next_gstate = 0
        else:
            current_gstate = 0
            return current_gstate

        if self.verbose:         
            print("this is current gstate: ",current_gstate)
            print("this is next gstate: ",next_gstate)

        if global_t < 100 and current_gstate > 1:
            print(self.growth_state_estimates[id_])
            raise OSError 
        
        current_descriptors = self.get_feature_descriptors(detections_t)
        if len(detections_t) > 0 and current_descriptors.size == 0:
            print("detections_t: ",detections_t)
            raise OSError 
        
        if current_gstate not in self.cone_feature_description_cache.keys() and id_ in self.cone_ids: 
            self.cone_feature_description_cache[current_gstate] = []
        elif current_gstate not in self.tree_feature_description_cache.keys() and id_ in self.tree_ids:  
            self.tree_feature_description_cache[current_gstate] = []

        #print("determining feature similarity ... ")
        if id_ in self.cone_ids: 
            if isinstance(self.cone_feature_description_cache[current_gstate],list): 
                 self.cone_feature_description_cache[current_gstate] = np.array(self.cone_feature_description_cache[current_gstate])
            if self.cone_feature_description_cache[current_gstate].size > 0:
                feature_similarity = self.determine_similarity(self.cone_feature_description_cache[current_gstate],current_descriptors) 
            else:
                feature_similarity = 0 
        elif id_ in self.tree_ids:
            if isinstance(self.tree_feature_description_cache[current_gstate],list): 
                self.tree_feature_description_cache[current_gstate] = np.array(self.tree_feature_description_cache[current_gstate])
            if self.tree_feature_description_cache[current_gstate].size > 0:
                feature_similarity = self.determine_similarity(self.tree_feature_description_cache[current_gstate],current_descriptors) 
            else:
                feature_similarity = 0 
        else:
            raise OSError 
        
        if self.verbose: 
            print("feature similarity: ",feature_similarity)

        if d_t <= T_nu: 
            if d_t < T_nu*.05:
                if self.verbose: 
                    print("d_t is much less than T_nu")
                #this is rachet ngl
                return current_gstate
            #print("the time for this growth state has not yet been exceeded")
            if self.hamming_similarity_val < feature_similarity: 
                if self.verbose: 
                    print("self.hamming_similarity_val:",self.hamming_similarity_val)
                    print("these features are different ... theres probably been a state change") 
                return next_gstate 
            else: 
                if self.verbose: 
                    print("the feature similarity is high!") 
                if feature_similarity < self.hamming_similarity_val * (1-self.growth_state_relevancy_level): 
                    #print("these are very similar, adding these descriptors!")
                    if id_ in self.cone_ids: 
                        if self.cone_feature_description_cache[current_gstate].size > 0:
                            self.cone_feature_description_cache[current_gstate] = np.vstack((self.cone_feature_description_cache[current_gstate],current_descriptors))
                        else:
                            self.cone_feature_description_cache[current_gstate] = current_descriptors 
                    elif id_ in self.tree_ids: 
                        if self.tree_feature_description_cache[current_gstate].size > 0: 
                            self.tree_feature_description_cache[current_gstate] = np.vstack((self.tree_feature_description_cache[current_gstate],current_descriptors))
                        else:
                            self.tree_feature_description_cache[current_gstate] = current_descriptors
                    else:
                        raise OSError 
                    return current_gstate 
                else:
                    #print("the features are different ... sample the gstate using the current time") 
                    if id_ in self.cone_ids: 
                        sampled_gstate = self.sample_gstate(id_,x=d_t/self.sim_length)
                    else:
                        sampled_gstate = self.sample_gstate(id_)
                    if sampled_gstate == 0:
                        if self.verbose: 
                            print("WARNING! sampling function thinks this clique should not persist!") 
                    return sampled_gstate 
        else:       
            if self.verbose: 
                print("the time in this growth state has been exceeded!") 
            if self.hamming_similarity_val < feature_similarity:
                #print("...however the features are quite different")
                if id_ in self.cone_ids: 
                    self.exceeded_Tnu_cone_tsteps += 1 
                    if 10 < self.exceeded_Tnu_cone_tsteps:
                        self.T_nu_cone -= self.T_nu_cone*0.05
                        self.exceeded_Tnu_cone_tsteps = 0
                else: 
                    self.exceeded_Tnu_tree_tsteps += 1 
                    if 10 < self.exceeded_Tnu_tree_tsteps: 
                        self.T_nu_tree -= self.T_nu_tree*0.05 
                        self.exceeded_Tnu_tree_tsteps = 0 
                return next_gstate
            else: 
                if self.verbose: 
                    print("the features are quite similar ...") 
                #increase the mean of the distribution for this clique state 
                if id_ in self.cone_ids: 
                    self.T_nu_cone += self.T_nu_cone*0.05
                else: 
                    self.T_nu_tree += self.T_nu_tree*0.05
                return current_gstate
            
    def sample_gstate(self,id_,x=None): 
        if id_ in self.cone_ids: 
            return cone_gstate_function(x)
        else:
            return tree_gstate_function(self.current_exp) 

    def get_feature_descriptors(self,current_observations):
        #print("getting feature descriptors ... this is current observations: ",current_observations)
        #current_observations is a list of dicts 
        current_feature_descriptors = []
        for obs in current_observations: 
            if "feature_des" in obs.keys(): 
                feature_descriptor = obs["feature_des"]
            else:
                if self.isCarla:
                    print("obs: ",obs)
                    raise OSError
                feature_descriptor = np.random.randint(0, 256, (32,), dtype=np.uint8) 
            current_feature_descriptors.append(feature_descriptor)
        current_feature_descriptors = np.array(current_feature_descriptors)
        return current_feature_descriptors 
    
    def determine_similarity(self,gstate_feature_descriptions,current_feature_descriptors): 
        if gstate_feature_descriptions.size > 0 and current_feature_descriptors.size > 0:
            hamming_ds = compute_hamming_distances(gstate_feature_descriptions.astype(int),current_feature_descriptors.astype(int))
            min_values = np.min(hamming_ds, axis=1); 
            similarity_val = np.mean(min_values)
        else: 
            similarity_val = self.hamming_similarity_val 
        return similarity_val 