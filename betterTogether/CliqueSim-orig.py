import numpy as np
import sys
from statsmodels.stats.weightstats import ztest as ztest
import JointPersistenceFilter
import pickle 

class clique_simulator():
    def __init__(self,results_dir,clique_features,total_experiments,sim_length,max_sim_length,negative_suppresion,P_Miss_detection,P_False_detection,confidence_range,lambda_u,
        acceptance_threshold,rejection_threshold,tree_ids,cone_ids):
        self.total_experiments 						= total_experiments
        self.sim_length        						= sim_length
        self.max_sim_length                         = max_sim_length
        self.negative_suppresion					= negative_suppresion
        self.P_Miss_detection					    = P_Miss_detection
        self.P_False_detection						= P_False_detection
        self.confidence_range                       = confidence_range
        self.clique_features                        = clique_features
        self.tracked_cliques                        = {}
        self.tree_survival                          = {}
        self.bad_tree_survival                      = {}
        self.data_association                      = {}
        for run in range(total_experiments):    
            self.data_association[run] = np.genfromtxt(results_dir + "data_association/experiment" + str(run) + "data_association.csv")
        self.cone_survival                          = {} 
        for c in self.clique_features.keys():
            if c not in self.tracked_cliques.keys():
                self.tracked_cliques[c] = JointPersistenceFilter.PersistenceFilterNd(lambda_u,num_features=len(self.clique_features[c]))
            if c in tree_ids:
                self.tree_survival[c] = total_experiments / 4
                self.bad_tree_survival[c] = total_experiments / 3
            else:
                self.cone_survival[c] = 2
        #survival time tuning 
        self.obsd_feats_cache						= {}
        self.feat_des_cache                         = {} 
        self.good_gstate_cache                      = {} 
        self.bad_gstate_cache						= {}
        for c in self.clique_features.keys():
            self.good_gstate_cache[c] = np.zeros((sim_length*total_experiments,))
            self.bad_gstate_cache[c] = np.zeros((sim_length*total_experiments,))
        self.bad_tree_survival_samples              = []
        self.tree_survival_samples                  = []
        self.cone_survival_samples                  = []  
        self.epsilon                                = .001
        self.all_tree_ids                           = tree_ids 
        self.all_cone_ids                           = cone_ids
        #logging 
        self.posteriors                             = {}
        self.independent_posteriors					= {} 
        self.tuned_posteriors						= {}
        self.acceptance_threshold                   = acceptance_threshold
        self.rejection_threshold                    = rejection_threshold

    '''
    def load(self,pik):
        with open(pik,"rb"):
            self.__dict__.update(pickle.loads(pik).__dict__)
        
    def save(self,sim_type):
        with open(sim_type + ".pickle","wb") as handle:
            return pickle.dump(self,handle,protocol=pickle.HIGHEST_PROTOCOL)
    '''

    def update(self,args):
        t = args[0]; detections_t = args[1]; tune = args[2];
        
        #print("detections_t: ",detections_t)
        observed_cliques = np.unique([x['clique_id'] for x in detections_t])

        detection_lists = {} 
        for c in observed_cliques:
            detection_lists[c] = [0 for x in self.clique_features[c].keys()]

        for el in detections_t:
            #print("this is el: ",el)
            if el['detection']:
                if el['clique_id'] in detection_lists.keys():
                   # print("el['feature_id']: ",el['feature_id'])
                    detection_lists[el['clique_id']][el['feature_id'] - 1] = 1 

        # remove bad single feature detections
        if self.negative_suppresion:
            for i,el in enumerate(detection_lists):
                #print("el: ",el)
                if np.sum(el) <= 1 and self.confidence_range < el['range']:
                    del detection_lists[i]
                    del observed_cliques[i]

        for c in observed_cliques:
        	self.tracked_cliques[c].update(detection_lists[c],t,self.P_Miss_detection,self.P_False_detection)

        for c in self.tracked_cliques.keys():
            if c not in self.posteriors.keys():
                #print("adding new posterior key: ",c)
                self.posteriors[c] = np.zeros((self.sim_length*self.total_experiments,))
                self.independent_posteriors[c] = np.zeros((self.sim_length*self.total_experiments,))
            self.posteriors[c][t] = self.tracked_cliques[c].predict_clique_likelihood(t)
            self.independent_posteriors[c][t] = self.tracked_cliques[c].predict(t,0)
            if tune:
                if c not in self.tuned_posteriors.keys():
                    self.tuned_posteriors[c] = np.zeros((self.sim_length*self.total_experiments,))
                self.tuned_posteriors[c][t] = self.tracked_cliques[c].predict_clique_likelihood(t)

        posteriors_t = [self.posteriors[c][t] for c in self.posteriors.keys()]
        if len([x for x in posteriors_t if x > 1]):
            normalized_posteriors = self.normalize(posteriors_t)
            for i,c in enumerate(self.posteriors.keys()):
                self.posteriors[c][t] = normalized_posteriors[i]

        if tune:
            self.tune_survival_time(t,detections_t)
            if not self.tuned_posteriors.keys() == self.posteriors.keys():
                raise OSError 
            tuned_posteriors_t = [self.tuned_posteriors[c][t] for c in self.tuned_posteriors.keys()]
            if len([x for x in tuned_posteriors_t if x > 1]):
                normalized_posteriors = self.normalize(tuned_posteriors_t)
                for i,c in enumerate(self.tuned_posteriors.keys()):
                    self.tuned_posteriors[c][t] = normalized_posteriors[i]

        persistent_obs = [] 
        for obs in detections_t:
        	c = obs['clique_id']
        	if tune:
        		if self.tuned_posteriors[c][t] > self.acceptance_threshold:
        			persistent_obs.append(obs)
        	else:
        		if self.posteriors[c][t] > self.acceptance_threshold:
        			persistent_obs.append(obs)

        return persistent_obs

    def normalize(self,posteriors):
        norm_factor = max(posteriors)
        normalized_posteriors = [x/norm_factor for x in posteriors]
        return normalized_posteriors 

    def get_associated_id(self,lm_id,run):
        current_data_association = self.data_association[run]
        lm_idx = [i for i,x in enumerate(current_data_association) if x[0] == lm_id]
        if len(lm_idx) == 0:
            #print("this lm id is not included in the data association... returning lm id")
            return lm_id

        lm_pos = current_data_association[lm_idx,1:][0]

        associated_id = lm_id
        for i in range(run):
            prev_data_association = self.data_association[i]
            #print("experiment0 data_association:", prev_data_association)
            #print("this is lm_pos: ",lm_pos)
            ds = [euclidean_distance(lm_pos,x) for x in prev_data_association[:,1:]]
            #print("this is ds: ",ds)
            #print("min_distance: ",min(ds))
            if min(ds) < 0.1:
                associated_lm_idx = np.argmin(ds)
                #print("this associated_lm_idx: ",associated_lm_idx)
                associated_id = int(prev_data_association[associated_lm_idx,0])
                if not isinstance(associated_id,int):
                    print("this is associated_id: ",associated_id)
                    raise OSError
                #else:
                #   print("this is associated_id: ",associated_id)
                return associated_id
            else:
                #print("all of these are hella far..")
                associated_id = int(lm_id)
                if not isinstance(associated_id,int):
                    #print("this is associated_id: ",associated_id)
                    raise OSError
                #else:
                #   print("there is not associated id... returning the lm id: ",associated_id)
        #print("at the bottom of get_associated_id: ",associated_id)
        return associated_id

    def cone_growth_model(self,observed,cone_id,t):
    	#self.max_experiments*0.75,(self.max_experiments*0.75)/3 #cones were generated according to a random normal distribution
        #print("cone survival keys",self.cone_survival.keys())
        for k in self.cone_survival.keys():
            if k not in self.clique_features.keys():
                raise OSError

        if cone_id not in self.cone_survival.keys():
            run = np.floor(t/self.sim_length)
            if self.get_associated_id(lm_id,run) not in self.cone_survival.keys():
                raise OSError
            else:
                cone_id = self.get_associated_id(lm_id,run)

        current_gstate = self.good_gstate_cache[cone_id][t]
        if current_gstate == 1:
            #is it likely this clique has died?
            if t < self.cone_survival[cone_id]*self.max_sim_length:
                if observed:
                    gstate = 1 
                else:
                    if np.random.rand() < self.P_Miss_detection:
                        gstate = 1 
                    else:
                        gstate = 0 
            else:
                last_diff_t = 0 
                for i in range(t):
                    if self.good_gstate_cache[cone_id][i] == 0:
                        if i > last_diff_t:
                            last_diff_t = i 
                if last_diff_t - t > self.cone_survival[cone_id]*self.max_sim_length: #the survival time has passed
                    if not observed:
                        gstate = 0
                    else: #it was observed 
                        if np.random.rand() < self.P_False_detection:
                            gstate = 0 #it was a false detection
                        else:
                            gstate = 1
                else: #the survival time hasnt been passed 
                    if observed:
                        gstate = 1 
                    else: #but it wasnt observed
                        if np.random.rand() < self.P_Miss_detection:
                            gstate = 1 #it was a missed detection
                        else:
                            gstate = 0 
        else:
            #did we make a mistake before?
            last_diff_t = 0 
            for i in range(t):
                if self.good_gstate_cache[cone_id][i] == 1: #find the last time it was supposedly alive 
                    if i > last_diff_t:
                        last_diff_t = i 

            last_diff_t0 = 0 
            for i in reversed(range(last_diff_t)):
                if self.good_gstate_cache[cone_id][i] == 0:
                    last_diff_t = i

            last_persistance_streak = last_diff_t - last_diff_t0 
            '''
            print("this is cone_id: ",cone_id)
            print("this is clique features keys: ",self.clique_features.keys())
            print("last_persistance_streak: ",last_persistance_streak)
            print("max_sim_length: ",self.max_sim_length)
            print("cone survival keys",self.cone_survival.keys())
            '''
            if last_persistance_streak < self.cone_survival[cone_id]*self.max_sim_length: 
                #its possible we made an error 
                if observed:
                    if np.random.rand() < self.P_False_detection:
                        gstate = 0 
                    else:
                        gstate = 1 
                else:
                    if np.random.rand() < self.P_Miss_detection:
                        gstate = 1 
                    else:
                        gstate = 0
            else:
                if observed: 
                    if np.random.rand() < self.P_False_detection:
                        gstate = 0
                    else:
                        gstate = 1
                else:
                    gstate = 0 

        return gstate

    def good_tree_model(self,t,id_):
        if id_ not in self.tree_survival.keys():
            run = np.floor(t/self.sim_length)
            id_ = self.get_associated_id(id_,run)
        #print('this is tree_survival: ',self.tree_survival)
        if t < self.tree_survival[id_]*self.max_sim_length:
            gstate = 1
        elif self.tree_survival[id_]*self.max_sim_length <= t < 2*self.tree_survival[id_]*self.max_sim_length:
            gstate = 2 
        elif 2*self.tree_survival[id_]*self.max_sim_length <= t < (4/3)*self.tree_survival[id_]*self.max_sim_length:
            gstate = 3
        elif (4/3)*self.tree_survival[id_]*self.max_sim_length <= t:
            gstate = 1
        return gstate

    def bad_tree_model(self,t,id_):
        if id_ not in self.tree_survival.keys():
            run = np.floor(t/self.sim_length)
            id_ = self.get_associated_id(id_,run)
        if t < self.bad_tree_survival[id_]*self.max_sim_length :
            gstate = 1
        elif self.bad_tree_survival[id_]*self.max_sim_length  <= t < (3/2)*self.bad_tree_survival[id_]*self.max_sim_length :
            gstate = 2 
        elif (3/2)*self.bad_tree_survival[id_]*self.max_sim_length <= t :
            gstate = 3
        return gstate

    def get_t_since_last_change(self,cache,c,t):
        previous_gstates = cache[c][1:t]
        tmp = [i for i,x in enumerate(previous_gstates) if x!=cache[c][t]]
        if len(tmp)>0:
            t_last_growth = max(tmp)
        else:
            t_last_growth = 0
        return t_last_growth

    def tune_survival_time(self,t,detections_t): 
        #print("this is detections_t: ",detections_t)
        for c in [x['clique_id'] for x in detections_t]:
            if c not in self.clique_features.keys():
                raise OSError

            new_posterior =  self.tuned_posteriors[c][t]
            
            reinit = False 

            detections = [x for x in detections_t if x['clique_id'] == c]
            if c not in self.obsd_feats_cache.keys():
                self.obsd_feats_cache[c] = []
            if c not in self.feat_des_cache.keys():
                self.feat_des_cache[c] = {} 

            if self.tuned_posteriors[c][-1] + self.epsilon < self.tuned_posteriors[c][-2]:
                #the posterior of an observed clique is dropping 
                #is there an occlusion?
                t_last_growth = self.get_t_since_last_change(self.good_gstate_cache,c,t)
                detections = [x for x in detections if x['clique_id'] == c]
                obsd_feat_ids = [x['feature_id'] for x in detections]
                #update the feature cache 
                for feat_id in obsd_feat_ids:
                    if feat_id not in self.obsd_feats_cache[c]:
                        self.obsd_feats_cache[c].append(feat_id)
                p = len(detections) / len(self.obsd_feats_cache[c])
                if p < 0.1:
                    #there was an occlusion reinit the posterior to 1 
                    new_posterior = 1
                    return new_posterior, reinit
                else:
                    #if it was a tree was there a change of season? 
                    if c in self.all_tree_ids:
                        estimated_gstate0 = self.good_tree_model(t,c)
                        estimated_gstate1 = self.bad_tree_model(t,c)
                        if estimated_gstate0 != self.good_gstate_cache[c][t]:
                            #there was just a change in the growth state do not reinit the EKF 
                            if self.check_change_feats(c,t,detections):
                                if len(self.tree_survival_samples) < 2:
                                    new_posterior = 1
                                    t_last_growth = self.get_t_since_last_change(self.good_gstate_cache,t)
                                    self.tree_survival_samples.append(t_last_growth)
                                    if estimated_gstate1 != self.good_gstate_cache[c][t]:
                                        t_last_growth = self.get_t_since_last_change(self.good_gstate_cache,t)
                                        new_posterior = 1
                                        self.bad_tree_survival_samples.append(t_last_growth)
                                else:
                                    #check if it fits with the distribution 
                                    t_last_growth = self.get_t_since_last_change(self.good_gstate_cache,t)
                                    tval, p = ztest(t_last_growth,self.tree_survival_samples)
                                    if p > 0.9:
                                        new_posterior = 1 
                                        self.tree_survival_samples.append(t_last_growth)
                                    tval, p = ztest(t_last_growth,self.bad_tree_survival_samples)
                                    if p > 0.9:
                                        new_posterior = 1 
                                        t_last_growth = self.get_t_since_last_change(self.bad_gstate_cache,t)
                                        self.bad_tree_survival_samples.append(t_last_growth)
                                #resample the survival times 
                                if p > 0.9:
                                    sigma = np.var(self.tree_survival_samples)
                                    mu = np.mean(self.tree_survival_samples)
                                    self.tree_survival[c] = np.random.normal(mu,sigma,1)
                                    sigma = np.var(self.bad_tree_survival_samples)
                                    mu = np.mean(self.bad_tree_survival_samples)
                                    self.bad_tree_survival[c] = np.random.normal(mu,sigma,1)
                                    self.good_gstate_cache[c][t] = self.good_tree_model(t,c) #correct the growth state using the new survival time estimate
                                    self.bad_gstate_cache[c][t] = self.bad_tree_model(t,c)
                                    return new_posterior, reinit 
                    else:
                        #if it was a cone is this the same cone as before or a different one 
                        estimated_gstate = self.cone_growth_model(True,c,t)
                        
                        if estimated_gstate != self.good_gstate_cache[c][t]:
                            if len(self.cone_survival_samples) < 2:
                                self.cone_survival_samples.append(t_last_growth)
                                new_posterior = 1 
                                reinit = True
                                return new_posterior, reinit 
                            else:
                                tval, p = ztest(t_last_growth,self.cone_survival_samples)
                                if p > 0.9:
                                    new_posterior = 1 
                                    self.cone_survival_samples.append(t_last_growth)
                                    sigma = np.var(self.cone_survival_samples)
                                    mu = np.mean(self.cone_survival_samples)
                                    self.cone_survival[c] = np.random.normal(mu,sigma,1)
                                    self.good_gstate_cache[c][t] = self.cone_growth_model(True,c,t)
                                    return new_posterior, reinit 
                if new_posterior == self.tuned_posteriors[c][t]:
                    #there was neither an occlusion nor has the season changed
                    #try changing the expected survival time 
                    if c in self.all_tree_ids:
                        if t_last_growth > self.tree_survival[c]:
                            self.tree_survival[c] -= self.sim_length
                        else:
                            self.tree_survival[c] += self.sim_length
                        if t_last_growth > self.bad_tree_survival[c]:
                            self.bad_tree_survival[c] -= self.sim_length
                        else:
                            self.bad_tree_survival[c] += self.sim_length
                    else:
                        if t_last_growth > self.cone_survival[c]:
                            self.cone_survival[c] -= self.sim_length
                        else:
                            self.cone_survival[c] += self.sim_length
            else: #the posterior isnt dropping :) 
                if new_posterior < self.rejection_threshold:
                    if c in self.all_tree_ids:
                        self.good_gstate_cache[c][t] = 0 
                        self.bad_gstate_cache[c][t] = 0 
                    else:
                        self.good_gstate_cache[c][t] = 0 
                else:
                    if c in self.all_tree_ids:
                        self.good_gstate_cache[c][t] = self.good_tree_model(t,c)
                        self.bad_gstate_cache[c][t] = self.bad_tree_model(t,c)
                    else:
                        self.good_gstate_cache[c][t] = self.cone_growth_model(True,c,t)
                gstate_estimate = self.good_gstate_cache[c][t]
                if gstate_estimate not in self.feat_des_cache[c].keys():
                    self.feat_des_cache[c][gstate_estimate] = []
                for obs in detections:
                    #print(self.feat_des_cache[c][gstate_estimate])
                    if len(self.feat_des_cache[c][gstate_estimate])>0:
                        if not any((self.feat_des_cache[c][gstate_estimate][:]==obs['feature_des']).all(1)):
                            self.feat_des_cache[c][gstate_estimate].append(obs['feature_des'])
                    else:
                        self.feat_des_cache[c][gstate_estimate].append(obs['feature_des'])

        for c in [x for x in self.tracked_cliques.keys() if x not in [x['clique_id'] for x in detections_t]]:
            if c not in self.obsd_feats_cache.keys():
                #this clique has never been observed - its probably not initted yet 
                self.good_gstate_cache[c][t] = 0
                if c in self.all_tree_ids:
                    self.bad_gstate_cache[c][t] = 0 
            else:
                if c in self.all_cone_ids:
                    self.good_gstate_cache[c][t] = self.cone_growth_model(False,c,t)
                if c in self.all_tree_ids:
                    self.good_gstate_cache[c][t] = self.good_tree_model(t,c)
                    self.bad_gstate_cache[c][t] = self.bad_tree_model(t,c)

    def check_change_feats(self,id_,t,detections):
        #want to determine if the features have changed significantly 
        diff_feats = True 
        prev_gstate_estimate = self.good_gstate_cache[c][t-1]
        prev_gstate_feats = self.feat_des_cache[id_][prev_gstate_estimate]
        current_feat_des = [x['feature_des'] for x in detections]
        for x in current_feat_des:
            if any((prev_gstate_feats == x).all(1)):
                diff_feats = False 
                return diff_feats

        return diff_feats