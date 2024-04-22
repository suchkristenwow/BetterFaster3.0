import numpy as np
from clique_utils import get_tracked_cliques, last_index_of_change
#, determine_similarity 
from JointPersistenceFilter import PersistenceFilter
import os 
import sys
sys.path.append("../sim_utils")
from utils import get_reinitted_id  
import pickle 

class clique_simulator():
    def __init__(self,**kwargs):
        self.P_Miss_detection = kwargs.get("P_Miss_detection")
        self.P_False_detection = kwargs.get("P_False_detection")
        self.acceptance_threshold = kwargs.get("acceptance_threshold")
        self.rejection_threshold = kwargs.get("rejection_threshold")
        self.min_feats = kwargs.get("min_feats")
        self.max_feats = kwargs.get("max_feats")
        all_clique_feats = kwargs.get("clique_features")
        self.all_clique_feats = all_clique_feats 
        self.confidence_range =  kwargs.get("confidence_range")
        self.tune_bool = kwargs.get("tune")
        self.observed_ids = kwargs.get("observed_clique_ids")
        if all_clique_feats is None:
            raise OSError 
        self.clique_features = all_clique_feats
        exp = kwargs.get("experiment_no")
        all_observations_dir = kwargs.get("observations") 
        self.results_dir = all_observations_dir
        all_observation_files = [x for x in os.listdir(all_observations_dir) if "observations" in x]
        exp_observation_files = [x for x in all_observation_files if "exp" + str(exp) in x]
        if len(exp_observation_files) != 1:
            print("exp_observation_files: ",exp_observation_files)
            raise OSError 
        else:
            with open(os.path.join(self.results_dir,exp_observation_files[0]),"rb") as handle:
                exp_observations = pickle.load(handle)
        self.observations = exp_observations
        self.sim_length = kwargs.get("sim_length")
        self.lambda_u = kwargs.get("lambda_u")
        data_association = kwargs.get("data_association")
        self.data_association = data_association
        n_experiments = kwargs.get("n_experiments")
        self.n_experiments = n_experiments
        self.tracked_cliques = get_tracked_cliques(exp,self.sim_length,self.lambda_u,all_clique_feats,all_observations_dir,self.min_feats,self.max_feats,data_association) 
        if len(self.tracked_cliques) ==1:
            raise OSError
        self.posteriors = {}
        self.growth_state_estimates = {} 
        for c in self.observed_ids:
            #print("self.sim_length: {}, n_experiments: {}".format(self.sim_length,n_experiments))
            self.posteriors[c] = np.zeros((self.sim_length*n_experiments,))
            self.growth_state_estimates[c] = np.zeros((self.sim_length*n_experiments,))
        self.negative_suppresion = False 
        self.current_exp = exp 
        #growth state estimation 
        self.T_nu_cone = kwargs.get("survival_time_cone")
        self.T_nu_tree = kwargs.get("survival_time_tree")
        self.historical_gstate_estimates = kwargs.get("previous_gstate_estimates")
        self.feature_description_cache = {}
        self.growth_state_relevancy_level = kwargs.get("growth_state_relevancy_level")

    def reinit_experiment(self,exp,observations): 
        self.current_exp = exp 
        self.observations = observations 
        new_tracked_cliques = get_tracked_cliques(exp,self.sim_length,self.lambda_u,self.all_clique_feats,self.results_dir,self.min_feats,self.max_feats,self.data_association) 
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
        #print("global_t: ",global_t)
        #print("init tsteps: ",[self.tracked_cliques[x].init_tstep for x in self.tracked_cliques.keys()])
        observed_clique_id = [x for x in self.tracked_cliques.keys() if self.tracked_cliques[x].init_tstep <= global_t] 
        #print("observed_clique_id: ",observed_clique_id)
        observed_posteriors = [self.posteriors[x][global_t] for x in observed_clique_id]
        #print("observed_posteriors: ",observed_posteriors)
        norm_factor = max(observed_posteriors)
        #print("normalization factor: ",norm_factor)
        normalized_posteriors = []
        for c in self.posteriors.keys(): 
            #print("self.posteriors[c].shape: ",self.posteriors[c].shape)
            if c not in observed_clique_id:
                normalized_posteriors.append(1.0)
            else:
                p_c = self.posteriors[c][global_t] 
                #print("p_c: ",p_c)
                p_c = p_c / norm_factor 
                if np.isnan(p_c):
                    raise OSError 
                elif np.isinf(p_c): 
                    raise OSError
                normalized_posteriors.append(p_c)
        return normalized_posteriors 
    
    def update(self,t,detections_t):
        observed_cliques = np.unique([x['clique_id'] for x in detections_t])
        #print("these are the observed cliques:",observed_cliques)

        detection_lists = {} 
        for c in observed_cliques:
            #detection_lists[c] = np.random.choice([0, 1], size=len(self.clique_features[c].keys()))
            if c not in self.clique_features.keys():
                #(all_data_associations,n,id_): 
                #print("get_reinitted_id... this is c:",c)
                c = get_reinitted_id(self.data_association,self.current_exp,c)
                #print("this is c now:",c)
            '''
            print("this is c: ",c)
            print("this is how many feature c has: ",len(self.clique_features[c].keys()))
            print("self.clique_features[c].keys(): ",self.clique_features[c].keys())
            '''
            detection_lists[c] = [1 for x in self.clique_features[c].keys()]

        for el in detections_t:
            #print("this is el: ",el)
            if el['detection']:
                if el['clique_id'] in detection_lists.keys():
                    if el['feature_id'] <= len(detection_lists[el['clique_id']]):
                        '''
                        print("el[clique_id]: ",el['clique_id'])
                        print("el['feature_id']: ",el['feature_id'])
                        print("len(detection_lists[el['clique_id']]):",len(detection_lists[el['clique_id']]))
                        '''
                        detection_lists[el['clique_id']][el['feature_id']-1] = 1 

        # remove bad single feature detections
        bad_single_feat_detections = []
        not_obsd_cliques = []
        if self.negative_suppresion:
            for i,k in enumerate(detection_lists.keys()):
                el = detections_t[i]
                if np.sum(detection_lists[k]) <= 1 and self.confidence_range < el['range']:
                    bad_single_feat_detections.append(i)
                    #del detection_lists[i]
                    not_obsd_cliques.append(observed_cliques[i])

        detection_lists = {key: value for key, value in detection_lists.items() if key not in bad_single_feat_detections}
        observed_cliques = [x for x in observed_cliques if x not in not_obsd_cliques]

        global_t = (self.current_exp - 1)*self.sim_length + t 

        for c in observed_cliques:    
            self.tracked_cliques[c].update(detection_lists[c],global_t,self.P_Miss_detection,self.P_False_detection)

        for c in self.tracked_cliques.keys():
            if c not in self.posteriors.keys():
                #print("adding new posterior key: ",c)
                self.posteriors[c] = np.zeros((self.sim_length*self.n_experiments,))
            #print("predicting clique likelihood of clique {}".format(c))
            if np.isnan(self.tracked_cliques[c].predict_clique_likelihood(t)):
                raise OSError 
            self.posteriors[c][global_t] = self.tracked_cliques[c].predict_clique_likelihood(t)
        
        posteriors_t = [self.posteriors[c][global_t] for c in self.posteriors.keys()]

        #print("this is posteriors_t before normalization:",posteriors_t)
        if len([x for x in posteriors_t if x > 1]):
            normalized_posteriors = self.normalize(global_t)
            #print("normalized posteriors: ",normalized_posteriors)
            if np.isnan(normalized_posteriors).any():
                raise OSError 
            for i,c in enumerate(self.posteriors.keys()):
                self.posteriors[c][global_t] = normalized_posteriors[i]
                if c in observed_cliques and self.posteriors[c][global_t] < 0.5: 
                    print("this is c:{} and the posterior: {}".format(c,self.posteriors[c][global_t]))
                    print("WARNING THIS POSTERIOR IS LOW EVEN THOUGH THE CLIQUE IS BEING OBSERVED?")
                    #raise OSError
        else:
            #print("no need to normalize...")
            normalized_posteriors = posteriors_t 
    
        #print("normalized posteriors: ",normalized_posteriors)    
        persistent_obs = [x for x in detections_t if self.posteriors[x['clique_id']][global_t] > self.acceptance_threshold] 

        #do growth state estimation 
        non_persistent_cliques = np.unique([x["clique_id"] for x in detections_t if self.posteriors[x["clique_id"]][global_t] <= self.rejection_threshold])  
        for c in non_persistent_cliques: 
            self.growth_state_estimates[c][global_t] = 0
        other_cliques = np.unique([x["clique_id"] for x in detections_t if self.posteriors[x["clique_id"]][global_t] > self.rejection_threshold])  
        for c in other_cliques:
            self.growth_state_estimates[c][global_t] = self.estimate_growth_state(t,c,detections_t) 

        return persistent_obs

    def estimate_growth_state(self,t,c,detections_t): 
        return 1 
    
    '''
    def estimate_growth_state(self,t,id_,detections_t): 
        #need to get time in the current growth state, then compare to T_nu_lmType  
        #self.comparison_growth_state_estimates[type_][id_] = np.zeros((self.sim_length,))
        if id_ in self.cone_ids:
            T_nu = self.T_nu_cone  
        elif id_ in self.tree_ids: 
            T_nu = self.T_nu_tree  
            
        print("this is the current exp:",self.current_exp)
        prev_gstates_id = []
        for x in sorted(self.historical_gstate_estimates): 
            gstate_estimates_x = self.historical_gstate_estimates[x]
            if id_ not in gstate_estimates_x.keys():
                # get_reinitted_id(self.data_association,self.current_exp,c)
                reinit_id = get_reinitted_id(self.data_association,self.current_exp,id_)
                if not reinit_id in gstate_estimates_x.keys():
                    print("this is id_: {} and this is current experiment: {}".format(id_,self.current_exp))
                    print("gstate_estimates.keys(): ",gstate_estimates_x)
                    print("reinit_id: ",reinit_id)
                    raise OSError 
                prev_gstates_id.append(gstate_estimates_x[reinit_id]) 
            else:
                prev_gstates_id.append(gstate_estimates_x[id_])  
        print("prev_gstates_id: ",prev_gstates_id)
        d_t = self.current_exp - last_index_of_change(prev_gstates_id)

        if t > 0:
            current_gstate = self.growth_state_estimates[id_][t-1]
        else:
            current_gstate = 1 
        
        #im just hard-coding this in idk 
        if id_ in self.tree_ids:
            if current_gstate == 0:
                #woops we thought this was dead... we probably missed up 
                return self.sample_gstate()
            elif 0 < current_gstate:
                if current_gstate + 1 <= 3:
                    current_gstate += 1
                else:
                    current_gstate = 1
            elif current_gstate < 0:
                raise OSError 
        elif id_ in self.cone_ids:
            if current_gstate == 0:
                next_gstate = 1 
            else:
                next_gstate = 0

        global_t = (self.current_exp - 1)*self.sim_length + t 
        feature_similarity = determine_similarity(self.feature_description_cache[current_gstate],detections_t)

        if T_nu < d_t: 
            # self.feature_description_cache 
            if feature_similarity < 1 - self.growth_state_relevancy_level: 
                return next_gstate 
            else: 
                if feature_similarity < self.growth_state_relevancy_level: 
                    return self.sample_gstate()
                elif self.growth_state_relevancy_level <= feature_similarity: 
                    #add t to clique state distribution
                    return current_gstate 
        else:
            if self.growth_state_relevancy_level < feature_similarity:
                #increase the mean of the distribution for this clique state 
                return current_gstate
            else: 
                return next_gstate
    '''
    