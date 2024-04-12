import numpy as np
from clique_utils import get_tracked_cliques
from JointPersistenceFilter import PersistenceFilter
import random 

import sys
sys.path.append("../sim_utils")
from utils import get_reinitted_id  

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
        all_observations = kwargs.get("observations")
        if all_observations is None:
            raise OSError 
        self.observations = all_observations
        self.sim_length = kwargs.get("sim_length")
        self.lambda_u = kwargs.get("lambda_u")
        data_association = kwargs.get("data_association")
        self.data_association = data_association
        exp = kwargs.get("experiment_no")
        n_experiments = kwargs.get("n_experiments")
        self.n_experiments = n_experiments
        self.tracked_cliques = get_tracked_cliques(exp,self.lambda_u,all_clique_feats,all_observations,self.min_feats,self.max_feats,data_association) 
        if len(self.tracked_cliques) ==1:
            raise OSError
        self.posteriors = {}
        self.growth_state_estimates = {} 
        for c in self.observed_ids:
            self.posteriors[c] = np.zeros((self.sim_length*n_experiments,))
            self.growth_state_estimates[c] = np.zeros((self.sim_length*n_experiments,))
        self.negative_suppresion = False 
        self.current_exp = exp 

    def reinit_experiment(self,exp,observations): 
        self.current_exp = exp 
        self.observations = observations 
        new_tracked_cliques = get_tracked_cliques(exp,self.lambda_u,self.all_clique_feats,observations,self.min_feats,self.max_feats,self.data_association) 
        print("new_tracked_cliques: ",new_tracked_cliques.keys())
        print("self.tracked_cliques.keys(): ",self.tracked_cliques.keys())
        for c in self.tracked_cliques.keys():
            old_clique_likelihood = self.tracked_cliques[c]._clique_likelihood 
            old_log_clique_evidence = self.tracked_cliques[c]._log_clique_evidence 
            self.tracked_cliques[c] = new_tracked_cliques[c]
            self.tracked_cliques[c]._clique_likelihood = old_clique_likelihood 
            self.tracked_cliques[c]._log_clique_evidence = old_log_clique_evidence

    def normalize(self,global_t):
        observed_clique_id = [x for x in self.tracked_cliques.keys() if self.tracked_cliques[x].init_tstep <= global_t]
        observed_posteriors = [self.posteriors[x][global_t] for x in observed_clique_id]
        norm_factor = max(observed_posteriors)
        normalized_posteriors = []
        for c in self.posteriors.keys(): 
            if c not in observed_clique_id:
                normalized_posteriors.append(1.0)
            else:
                p_c = self.posteriors[c][global_t] 
                p_c = p_c / norm_factor 
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
                print("get_reinitted_id... this is c:",c)
                c = get_reinitted_id(self.data_association,self.current_exp,c)
                print("this is c now:",c)
            print("this is c: ",c)
            print("this is how many feature c has: ",len(self.clique_features[c].keys()))
            detection_lists[c] = [1 for x in self.clique_features[c].keys()]

        for el in detections_t:
            if el['detection']:
                if el['clique_id'] in detection_lists.keys():
                    print("detection_lists.keys():",detection_lists.keys()) 
                    print("el[clique_id]: ",el['clique_id'])
                    print("el['feature_id']: ",el['feature_id'])
                    print("len(detection_lists[el['clique_id']]):",len(detection_lists[el['clique_id']]))
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

        for c in observed_cliques:
            #print("updating clique {}".format(c))
            #print("detection_lists[c]: ",detection_lists[c])
            self.tracked_cliques[c].update(detection_lists[c],t,self.P_Miss_detection,self.P_False_detection)

        global_t = (self.current_exp - 1)*self.sim_length + t 

        for c in self.tracked_cliques.keys():
            if c not in self.posteriors.keys():
                #print("adding new posterior key: ",c)
                self.posteriors[c] = np.zeros((self.sim_length*self.n_experiments,))
            #print("predicting clique likelihood of clique {}".format(c))
            self.posteriors[c][global_t] = self.tracked_cliques[c].predict_clique_likelihood(t)
        
        posteriors_t = [self.posteriors[c][global_t] for c in self.posteriors.keys()]

        print("posteriors: ",posteriors_t)
        '''
        print("this is posteriors_t before normalization:",posteriors_t)
        if len([x for x in posteriors_t if x > 1]):
            normalized_posteriors = self.normalize(global_t)
            print("normalized posteriors: ",normalized_posteriors)
            for i,c in enumerate(self.posteriors.keys()):
                self.posteriors[c][global_t] = normalized_posteriors[i]
                if c in observed_cliques and self.posteriors[c][global_t] < 0.5: 
                    print("this is c:{} and the posterior: {}".format(c,self.posteriors[c][global_t]))
                    print("WARNING THIS POSTERIOR IS LOW EVEN THOUGH THE CLIQUE IS BEING OBSERVED?")
                    raise OSError
        else:
            print("no need to normalize...")
            normalized_posteriors = posteriors_t 
        '''
        
        persistent_obs = [x for x in detections_t if self.posteriors[x['clique_id']][global_t] > self.acceptance_threshold] 

        #do growth state estimation 
        non_persistent_cliques = np.unique([x["clique_id"] for x in detections_t if self.posteriors[x["clique_id"]][global_t] <= self.rejection_threshold])  
        for c in non_persistent_cliques: 
            self.growth_state_estimates[c][global_t] = 0
        other_cliques = np.unique([x["clique_id"] for x in detections_t if self.posteriors[x["clique_id"]][global_t] > self.rejection_threshold])  
        for c in other_cliques:
            self.growth_state_estimates[c][global_t] = self.estimate_growth_state(c) 

        return persistent_obs

    def estimate_growth_state(self,id_): 
        return 1 