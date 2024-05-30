import numpy as np 
from .clique_utils import get_tracked_cliques, cone_gstate_function, tree_gstate_function 
from betterFaster.sim_utils.utils import get_reinitted_id 
from scipy.spatial.distance import cdist
import os 
import threading 

def compute_hamming_distances(descriptors1, descriptors2):
    # Convert binary arrays to integers for Hamming distance calculation
    d1 = np.packbits(descriptors1, axis=-1)
    d2 = np.packbits(descriptors2, axis=-1)
    
    # Compute Hamming distances
    hamming_distances = cdist(d1, d2, metric='hamming')
    
    # Since Hamming distance from cdist gives the fraction of differing bits,
    # we multiply by the number of bits per descriptor to get actual bit differences
    bit_length = descriptors1.shape[1] * 8  # 32 bytes * 8 bits
    hamming_distances *= bit_length
    
    return hamming_distances

class clique_simulator(): 
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
        self.lambda_u = parameters["betterTogether"]["lambda_u"]
        self.all_data_associations = kwargs.get("data_association")
        self.n_experiments = parameters["experiments"]
        self.tracked_cliques = get_tracked_cliques(self.current_exp,parameters,self.all_data_associations[self.current_exp + 1],self.clique_features)
        self.posteriors = {}
        self.growth_state_estimates = {} 
        for c in self.observed_ids:
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

    def reinit_experiment(self,exp,observations): 
        print("reinitting experiment...")
        self.current_exp = exp 
        self.observations = observations 
        #new_tracked_cliques = get_tracked_cliques(exp,self.sim_length,self.lambda_u,self.clique_features,self.results_dir,self.min_feats,self.max_feats,self.data_association) 
        new_tracked_cliques = get_tracked_cliques(exp,self.parameters,self.all_data_associations[self.current_exp + 1],self.clique_features)
        for c in self.tracked_cliques.keys():
            old_clique_likelihood = self.tracked_cliques[c]._clique_likelihood 
            old_log_clique_evidence = self.tracked_cliques[c]._log_clique_evidence 
            old_last_observation_time = self.tracked_cliques[c]._last_observation_time 
            old_log_likelihood = self.tracked_cliques[c]._log_likelihood 
            old_log_clique_lower_evidence_sum = self.tracked_cliques[c]._log_clique_lower_evidence_sum
            self.tracked_cliques[c] = new_tracked_cliques[c]
            self.tracked_cliques[c]._clique_likelihood = old_clique_likelihood 
            self.tracked_cliques[c]._log_clique_evidence = old_log_clique_evidence 
            self.tracked_cliques[c]._last_observation_time = old_last_observation_time
            self.tracked_cliques[c]._log_likelihood = old_log_likelihood
            self.tracked_cliques[c]._log_clique_lower_evidence_sum = old_log_clique_lower_evidence_sum 

    def normalize(self,global_t):
        observed_clique_id = [x for x in self.tracked_cliques.keys() if self.tracked_cliques[x].init_tstep <= global_t] 
        #print("these should be the cliques we already observed: ",observed_clique_id)

        observed_posteriors = [self.posteriors[x][global_t] for x in observed_clique_id]
        #print("observed_posteriors: ",observed_posteriors)

        norm_factor = max(observed_posteriors)

        normalized_posteriors = []
        for c in self.posteriors.keys(): 
            if c not in observed_clique_id:
                #we havent observed this yet 
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
                normalized_posteriors.append(p_c)
        return normalized_posteriors 
    
    def get_detection_lists(self,detections_t,observed_cliques): 
        detection_lists = {} 
        for c in observed_cliques:
            #detection_lists[c] = np.random.choice([0, 1], size=len(self.clique_features[c].keys()))
            if c not in self.clique_features.keys():
                #(all_data_associations,n,id_): 
                #print("get_reinitted_id... this is c:",c)
                c = get_reinitted_id(self.all_data_associations,self.current_exp,c)
            print("WARNING SETTING ALL FEATURES TO DETECTED...")
            detection_lists[c] = [1 for _ in self.clique_features[c].keys()]
            #detection_lists[c] = [0 for _ in self.clique_features[c].keys()]
        '''  
        for el in detections_t:
            if el['detection']:
                if el['clique_id'] in detection_lists.keys():
                    if el['feature_id']-1 < len(detection_lists[el['clique_id']]) or np.random.rand() < self.P_Miss_detection:
                        detection_lists[el['clique_id']][el['feature_id']-1] = 1 
                    else:
                        if el['feature_id'] > len(detection_lists[el['clique_id']]):
                            print("wtf")
                            print("el[feature_id]: ",el['feature_id'])
                            print("len(detection_lists[el['clique_id']]): ",len(detection_lists[el['clique_id']]))
                            raise OSError 
      
        for id_ in detection_lists.keys():
            total_obsd_feats_id = sum(detection_lists[id_])
            print("observed {} out of {} features for this clique: {}".format(total_obsd_feats_id,len(detection_lists[id_]),id_)) 
        '''
        return detection_lists

    '''
    def remove_bad_single_feat_detections(self,detection_lists,observed_cliques,detections_t):
        # remove bad single feature detections
        bad_single_feat_detections = []
        not_obsd_cliques = []
        #if self.negative_suppresion:
        for i,k in enumerate(detection_lists.keys()):
            el = detections_t[i]
            if np.sum(detection_lists[k]) <= 1 and self.confidence_range < el['range']:
                bad_single_feat_detections.append(i)
                not_obsd_cliques.append(observed_cliques[i])

        detection_lists = {key: value for key, value in detection_lists.items() if key not in bad_single_feat_detections}
        observed_cliques = [x for x in observed_cliques if x not in not_obsd_cliques]
        return detection_lists, observed_cliques
    '''

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
        #print("entering clique update...")
        observed_cliques = np.unique([x['clique_id'] for x in detections_t])
        #print("observed these cliques this timestep: ",observed_cliques)
        self.ever_obsd.extend([x for x in observed_cliques if x not in self.ever_obsd])

        #Get detection lists 
        detection_lists = self.get_detection_lists(detections_t,observed_cliques)

        #Remove bad single feat detections
        detection_lists,observed_cliques = self.remove_bad_single_feat_detections(detection_lists,observed_cliques,detections_t)

        global_t = self.current_exp*self.sim_length + t 

        if global_t < 0:
            raise OSError
        
        #Update observed cliques 
        '''
        for c in observed_cliques:    
            print("updating clique {} ..................................".format(c))
            self.tracked_cliques[c].update(detection_lists[c],global_t,self.P_Miss_detection,self.P_False_detection)
        '''
        threads = []
        for c in observed_cliques:
            thread_c = threading.Thread(target=self.parallel_update_helper, args=(c,detection_lists[c],global_t))
            threads.append(thread_c)
            thread_c.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        #predict clique likelihoods
        for c in self.tracked_cliques.keys():
            if c not in self.posteriors.keys():
                #print("adding new posterior key: ",c)
                self.posteriors[c] = np.zeros((self.sim_length*self.n_experiments,))
            #print("predicting clique likelihood of clique {}".format(c))
            posterior_c = self.tracked_cliques[c].predict_clique_likelihood(t)
            if np.isnan(posterior_c):
                print("clique likelihood is nan: ",)
                raise OSError 
            self.posteriors[c][global_t] = posterior_c
            #print("the posterior of clique {} is {}".format(c,self.posteriors[c][global_t]))

        #normalize posteriors 
        posteriors_t = [self.posteriors[c][global_t] for c in self.posteriors.keys()]
        #print("this is posteriors_t before normalization:",posteriors_t)
        if len([x for x in posteriors_t if x > 1]):
            normalized_posteriors = self.normalize(global_t)
            if np.isnan(normalized_posteriors).any():
                raise OSError 
            posteriors_t = normalized_posteriors 
            #print("this is posteriors_t after normalization: ",posteriors_t)

        for i,c in enumerate(self.posteriors.keys()):
            self.posteriors[c][global_t] = posteriors_t[i]
            if c in observed_cliques and self.posteriors[c][global_t] < 0.1:  
                print("this is c:{} and the posterior: {}".format(c,self.posteriors[c][global_t]))
                print("WHY IS THIS POSTERIOR SO LOW LOW EVEN THOUGH THE CLIQUE IS BEING OBSERVED?")
            
        persistent_obs = [x for x in detections_t if self.posteriors[x['clique_id']][global_t] > self.acceptance_threshold] 
    
        #do growth state estimation 
        non_persistent_cliques = np.unique([x["clique_id"] for x in detections_t if self.posteriors[x["clique_id"]][global_t] <= self.rejection_threshold])  
        for c in non_persistent_cliques: 
            #print("clique c: {} posterior is below the rejection threshold!".format(c))
            self.growth_state_estimates[c][global_t] = 0

        #persisting_cliques = np.unique([x["clique_id"] for x in detections_t if self.posteriors[x["clique_id"]][global_t] > self.rejection_threshold])  

        for c in self.posteriors.keys():
            estimated_gstate = self.estimate_growth_state(t,c,detections_t,self.posteriors[c][global_t])
            '''
            if estimated_gstate == 0:
                print("WARNING: GSTATE ESTIMATION EXPECTS THIS CLIQUE TO NOT PERSIST")
            '''
            if estimated_gstate > 1 and global_t < 10:
                print("growth state estimation is out of control")
                raise OSError 
            self.growth_state_estimates[c][global_t] = estimated_gstate

        if self.tune_bool: 
            persistent_obs = []
            for c in self.growth_state_estimates.keys(): 
                if self.growth_state_estimates[c][global_t] != 0: 
                    persistent_obs.extend([x for x in detections_t if get_reinitted_id(self.all_data_associations,self.current_exp,x['clique_id']) == c])

        return persistent_obs

    def last_index_of_change(self,id_,t,prev_gstates_id):
        global_t = self.current_exp * self.sim_length + t 
        #print("Trying to find out how long this clique has been in this clique state")
        if t > 0:
            if id_ not in self.growth_state_estimates.keys(): 
                reinit_id = get_reinitted_id(self.all_data_associations,self.current_exp,id_,optional_exp=0)
                if reinit_id not in self.growth_state_estimates.keys():
                    reinit_id = get_reinitted_id(self.all_data_associations,self.current_exp,id_,optional_exp=self.current_exp)
                    if reinit_id not in self.growth_state_estimates.keys():
                        raise OSError 
                id_ = reinit_id 

            current_gstate = self.growth_state_estimates[id_][t-1]
        else:
            current_gstate = 1 
        #print("this is current_gstate: ",current_gstate)

        if not isinstance(self.all_data_associations,dict): 
            print("self.all_data_associations: ",self.all_data_associations)
            raise OSError 
        
        if not id_ in self.cone_ids and id_ not in self.tree_ids: 
            orig_id = id_ 
            id_ = get_reinitted_id(self.all_data_associations,self.current_exp,id_,optional_exp=self.current_exp) 
            if not id_ in self.cone_ids and id_ not in self.tree_ids:  
                print("self.all_data_associations: ",self.all_data_associations)
                print("self.current_exp:",self.current_exp)
                print("orig_id: ",orig_id) 
                print("id_: ",id_) 
                print("self.cone_ids: ",self.cone_ids)
                print("self.tree_ids: ",self.tree_ids)
                raise OSError 
            
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
            raise OSError 
        
        #print("this is prev_gstate: ",prev_gstate)
        if prev_gstate in prev_gstates_id: 
            for i in range(len(prev_gstates_id) - 1, -1, -1):
                if prev_gstates_id[i] == prev_gstate:
                    #print("This is i: ",i)
                    if i > global_t:
                        raise OSError 
                    return i
        else: 
            return 0  # Return 0 if the value is not found

    def estimate_growth_state(self,t,id_,detections_t,posterior): 
        #print("estimating growth state of clique {}....".format(id_))
        #need to get time in the current growth state, then compare to T_nu_lmType  

        global_t = self.current_exp*self.sim_length + t 

        if not id_ in self.tree_ids and id_ not in self.cone_ids: 
            orig_id = id_ 
            id_ = get_reinitted_id(self.all_data_associations,self.current_exp,id_,self.current_exp) 
        
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
            elif orig_id in self.growth_state_estimates.keys():  
                prev_gstates_id = self.growth_state_estimates[orig_id][:global_t]
            else:
                print("orig_id: {}, id_:{}".format(orig_id,id_))
                print("self.growth_state_estimates.keys():",self.growth_state_estimates.keys())
                raise OSError 
            if self.last_index_of_change(id_,t,prev_gstates_id) > global_t:
                print("self.last_index_of_change: {}, global_t: {}".format(self.last_index_of_change(id_,t,prev_gstates_id),global_t))
                print("this isnt possible")
                raise OSError 
            #print("self.current_exp*self.sim_length: {}, self.last_index_of_change: {}".format((self.current_exp)*self.sim_length, self.last_index_of_change(id_,t,prev_gstates_id)))
            d_t = global_t - self.last_index_of_change(id_,t,prev_gstates_id)
    
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

        if id_ in self.tree_ids:
            if current_gstate == 0 and posterior > self.acceptance_threshold:
                #print("woops we thought this was dead... we probably missed up bc the posterior is really high") 
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
            # c = get_reinitted_id(self.all_data_associations,self.current_exp,c) 
            reinit_id = get_reinitted_id(self.all_data_associations,self.current_exp,id_,self.current_exp)
            print("this is neither a tree or a cone:",id_)
            print("this is reinit_id: ",reinit_id)
            print("self.cone_ids: ",self.cone_ids)
            print("self.tree_ids: ",self.tree_ids)
            raise OSError 
        
        #print("this is current gstate: ",current_gstate)
        #print("this is next gstate: ",next_gstate)

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
        
        #Note: LOWER feature_similarity value implies 
        #print("this is feature_similarity: ",feature_similarity)
        #print("this is d_t: {} and T_nu: {}".format(d_t,T_nu))

        if d_t <= T_nu: 
            if d_t < T_nu*.05:
                #this is rachet ngl
                return current_gstate
            #print("the time for this growth state has not yet been exceeded")
            if self.hamming_similarity_val < feature_similarity: 
                #print("self.hamming_similarity_val:",self.hamming_similarity_val)
                #print("these features are different ... theres probably been a state change") 
                return next_gstate 
            else: 
                # the feature similarity is high
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
                    '''
                    if sampled_gstate == 0:
                        print("WARNING! sampling function thinks this clique should not persist!") 
                    '''
                    return sampled_gstate 
        else:       
            #print("the time in this growth state has been exceeded!") 
            if self.hamming_similarity_val < feature_similarity:
                #however the features are quite different 
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
                #print("the features are quite similar ...") 
                #increase the mean of the distribution for this clique state 
                if id_ in self.cone_ids: 
                    self.T_nu_cone += self.T_nu_cone*0.05
                else: 
                    self.T_nu_tree += self.T_nu_tree*0.05
                return current_gstate
            
    def sample_gstate(self,id_,x=None): 
        #print("sampling gstate...")
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
                '''
                if not already_printed:
                    print("WARNING THERE IS NO FEATURE DESCRIPTOR: THIS IS THE MADE UP DATASET")
                    already_printed = True 
                '''
                feature_descriptor = np.random.randint(0, 256, (32,), dtype=np.uint8) 
            current_feature_descriptors.append(feature_descriptor)
        current_feature_descriptors = np.array(current_feature_descriptors)
        return current_feature_descriptors 
    
    def determine_similarity(self,gstate_feature_descriptions,current_feature_descriptors): 
        #print("gstate_feature_descriptions.size:",gstate_feature_descriptions.size)
        #print("current_feature_descriptors.size:",current_feature_descriptors.size)
        if gstate_feature_descriptions.size > 0 and current_feature_descriptors.size > 0:
            hamming_ds = compute_hamming_distances(gstate_feature_descriptions,current_feature_descriptors)
            min_values = np.min(hamming_ds, axis=1); 
            similarity_val = np.mean(min_values)
        else: 
            similarity_val = self.hamming_similarity_val 
        return similarity_val 