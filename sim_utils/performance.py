
import sys
sys.path.append("../betterTogether")
#from clique_utils import get_gt_clique_persistence
import numpy as np 
import os 
from utils import get_sim_length, get_reinitted_id

class PerformanceTracker:
    def __init__(self,n_experiments,data_associations=None):
        if data_associations is None:
            all_results_dir = "/media/arpg/easystore1/BetterFaster/kitti_carla_simulator/"
            if not os.path.exists(os.path.join(all_results_dir,"exp_results")): 
                all_results_dir = "/media/arpg/easystore/BetterFaster/kitti_carla_simulator/"
                if not os.path.exists(os.path.join(all_results_dir,"exp_results")):
                    raise OSError
            results_dir = os.path.join(all_results_dir,"exp_results")
        else:
            results_dir = "sim_utils/fake_data"

        self.results_dir = results_dir
        self.data_association = {} 
        self.gt_trees = {} 
        self.gt_cones = {} 

        for exp in range(n_experiments):
            #extract ground truth landmarks 
            cone_ids_file = os.path.join(results_dir,"cone_ids/experiment"+str(exp+1)+"cone_ids_.txt")
            # Read the contents of the text file
            with open(cone_ids_file, 'r') as file:
                lines = file.readlines()
            cone_ids = [int(line.strip()) for line in lines]

            # Convert each line to an integer and store in a list
            tree_ids_file = os.path.join(results_dir,"tree_ids/experiment"+str(exp+1)+"tree_ids_.txt")
            # Read the contents of the text file
            with open(tree_ids_file, 'r') as file:
                lines = file.readlines()
            # Convert each line to an integer and store in a list
            tree_ids = [int(line.strip()) for line in lines]

            data_association_path = os.path.join(results_dir,"data_association/experiment"+str(exp+1)+"data_association.csv")
            if os.path.exists(data_association_path) and data_associations is None:
                exp_data_association = np.genfromtxt(data_association_path,delimiter=" ") 
            else:
                data_association_path = "/home/kristen/BetterFaster3.0/sim_utils/fake_data/data_associations"
                exp_data_association = np.genfromtxt(os.path.join(data_association_path,"exp"+str(exp)+"_data_association.csv"))

            self.data_association[exp + 1]  = exp_data_association 
            self.gt_trees[exp + 1] = [x for x in exp_data_association if x[0] in tree_ids]
            self.gt_cones[exp + 1]  = [x for x in exp_data_association if x[0] in cone_ids]
            self.gt_cones[exp + 1]  = np.array(self.gt_cones[exp + 1])
            self.gt_trees[exp + 1]  = np.array(self.gt_trees[exp + 1])

        sim_length = get_sim_length(results_dir,n_experiments)
        self.gt_clique_persistence = {}
        self.all_data_associations = {}
        self.n_experiments = n_experiments
        print("in performance... getting data associations....")
        for n in range(1,n_experiments+1):
            print("this is exp no:{}...".format(n))
            if not self.results_dir is None:
                #dir_n = os.path.join(self.results_dir,"exp"+str(n)+"_results")
                data_associations_path = os.path.join(results_dir,"data_association/experiment"+str(n)+"data_association.csv")
                if not os.path.exists(data_associations_path):
                    data_associations_path = os.path.join(results_dir,"data_associations/exp"+str(n-1)+"_data_association.csv")
                    if not os.path.exists(data_associations_path):
                        print("data_associations_path:",data_associations_path)
                        raise OSError
                print("data_associations_path: ",data_associations_path)
                data_associations = np.genfromtxt(data_associations_path)
                if np.isnan(data_associations[:,0].any()):
                    data_associations = np.genfromtxt(data_associations_path,delimiter=" ")
                    print("this is data_associations: ",data_associations)
            else:
                if data_associations is None:
                    raise OSError
            if np.isnan(data_associations).any():
                raise OSError
            else:
                self.all_data_associations[n] = data_associations
                #print("assigning data_associations to self.all_data_associations... this is n:",n)
            #sanity check
            print("checking the data associations....")
            for x in self.all_data_associations.keys():
                #print("checking what we've set... this is n:",x)
                tmp = self.all_data_associations[x] 
                print("tmp:",tmp)
                if np.isnan(tmp).any():
                    raise OSError
            exp_clique_ids = data_associations[:,0]
            print("exp_clique_ids: ",exp_clique_ids)
            if n > 1 and self.results_dir is not None:
                for i,id_ in enumerate(exp_clique_ids):
                    #get_reinitted_id(all_data_associations,n,id_)
                    orig_id = get_reinitted_id(self.all_data_associations,n,id_) #this should be the original clique id
                    #print("this is orig_id: ",orig_id)
                    if not orig_id is None:
                        exp_clique_ids[i] = orig_id
            self.gt_clique_persistence[n] = exp_clique_ids
            #print("this is all_data_associations: ",self.all_data_associations[n]) 
        self.acceptance_threshold = None 
        self.rejection_threshold = None 
        self.posteriors = None 
        self.particles = None 
        self.best_traj_estimate = np.zeros((sim_length,3))
        self.sim_length = sim_length
        self.current_exp = None  
        self.best_landmark_estimates = {}
        self.accuracy = {}
        self.accuracy["true_positive"] = 0
        self.accuracy["true_negative"] = 0
        self.accuracy["false_positive"] = 0
        self.accuracy["false_negative"] = 0
        self.clique_inst = None 
        self.slam_inst = None 
        self.comparison_particles = {}
        self.comparison_sim_instance = {} 
        self.comparison_slam_instance = {}
        self.untuned_posteriors = None 
        self.comparison_accuracies = {} 
        self.comparison_best_traj_estimates = {}
        self.comparison_best_landmark_estimates = {}
        self.growth_state_estimates = None 
        self.comparison_growth_state_estimates = {} 

    '''
        def get_reinitted_ids(self,n,id_): 
        #want to see if this landmark existed in experiments before this 
        print("getting reinitted id.... this is id_:",id_)
        idx = np.where(self.all_data_associations[n][:,0] == id_)
        lm_id_pos = self.all_data_associations[n][idx,1:]
        print("lm_id_pos: ",lm_id_pos)
        reinitted_id = None 
        i = n - 1
        while 1 <= i:
            tmp = self.all_data_associations[i] 
            lms_i = tmp[:,1:]
            row_idx = np.where(np.all(lms_i == lm_id_pos, axis=1))[0]
            if row_idx.size > 0:
                print("this lm existed in the previous experiment")
                reinitted_id = self.all_data_associations[i][row_idx,0]
                return reinitted_id
            else:
                print("this lm did not exist in this experiment") 
                i -= 1
                break
    '''

    def init_new_experiment(self,experiment_no,clique,slam,compare_betterTogether,compare_multiMap,compare_vanilla):
        self.clique_inst = clique 
        self.slam_inst = slam 
        self.acceptance_threshold = clique.acceptance_threshold
        self.rejection_threshold = clique.rejection_threshold
        self.posteriors = clique.posteriors
        self.particles = slam.particles 
        self.current_exp = experiment_no
        #self.clique_sim = clique 
        #self.BF_SLAM = slam 

    def init_new_comparison(self,type_,sim,slam): 
        self.comparison_sim_instance[type_] = sim 
        self.comparison_slam_instance[type_] = slam 
        if "untuned" in type_:
            self.untuned_posteriors = sim.posteriors
        self.comparison_particles[type_] = slam.particles 

    def update_clique_posteriors(self,t,clique,type_=None):
        if type_ is None:
            self.clique_inst = clique 
            self.posteriors = clique.posteriors
            self.growth_state_estimates = clique.growth_state_estimates 
        else:
            self.comparison_sim_instance[type_] = clique 
            self.comparison_particles[type_] = clique.posteriors 
            self.comparison_growth_state_estimates[type_] = clique.growth_state_estimates 

        #want to update accuracy of persistence
        for c in clique.posteriors.keys():
            posterior_c = clique.posteriors[c][t]
            #get_reinitted_id(all_data_associations,n,id_)
            reinit_id = get_reinitted_id(self.all_data_associations,self.current_exp,c)
            if reinit_id is not None: 
                id_ = reinit_id
            else:
                id_ = c 
            if type_ is None:
                if id_ in self.gt_clique_persistence[self.current_exp]: 
                    #this clique is persisting
                    if posterior_c > self.acceptance_threshold:
                        #true positive
                        self.accuracy["true_positive"] += 1
                    else:
                        #false negative
                        self.accuracy["false_negative"] += 1
                else:
                    #this clique is not persisting
                    if posterior_c < self.rejection_threshold: 
                        #true negative
                        self.accuracy["true_negative"] +=1 
                    else:
                        #false positive
                        self.accuracy["false_positive"] += 1 
            else: 
                if not type_ in self.comparison_accuracies.keys(): 
                    self.comparison_accuracies[type_] = {} 
                    self.comparison_accuracies[type_]["true_positive"] = 0
                    self.comparison_accuracies[type_]["false_positive"] = 0
                    self.comparison_accuracies[type_]["true_negative"] = 0
                    self.comparison_accuracies[type_]["false_negative"] = 0

                if id_ in self.gt_clique_persistence[self.current_exp]: 
                    #this clique is persisting
                    if posterior_c > self.acceptance_threshold:
                        #true positive
                        self.comparison_accuracies[type_]["true_positive"] += 1
                    else:
                        #false negative
                        self.comparison_accuracies[type_]["false_negative"] += 1
                else:
                    #this clique is not persisting
                    if posterior_c < self.rejection_threshold: 
                        #true negative
                        self.comparison_accuracies[type_]["true_negative"] +=1 
                    else:
                        #false positive
                        self.comparison_accuracies[type_]["false_positive"] += 1 
        
                #Do growth state estimate 
                if type_ not in self.comparison_growth_state_estimates.keys():
                    self.comparison_growth_state_estimates[type_] = {} 
                if id_ not in self.comparison_growth_state_estimates[type_].keys():
                    self.comparison_growth_state_estimates[type_][id_] = np.zeros((self.sim_length,))

                if c in self.gt_trees[self.current_exp]: 
                    if self.current_exp < int(self.n_experiments*.25): 
                        #season is winter
                        if id_ in self.gt_clique_persistence[self.current_exp]:
                            gt_gstate = 1 
                        else:
                            gt_gstate = 0 
                    elif int(.25*self.n_experiments) <= self.current_exp < int(self.n_experiments*.5):
                        #season is summer 
                        if id_ in self.gt_clique_persistence[self.current_exp]:
                            gt_gstate = 2 
                        else:
                            gt_gstate = 0
                    elif int(self.n_experiments*.5) <= self.current_exp < int(self.n_experiments*.75): 
                        #season is fall 
                        if id_ in self.gt_clique_persistence[self.current_exp]:
                            gt_gstate = 3 
                        else: 
                            gt_gstate = 0
                    else:
                        #season is winter
                        if id_ in self.gt_clique_persistence[self.current_exp]: 
                            gt_gstate = 1 
                        else: 
                            gt_gstate = 0 
                else: 
                    if id_ in self.gt_clique_persistence[self.current_exp]:
                        gt_gstate = 1
                    else:
                        gt_gstate = 0  
                
                if clique.growth_state_estimates[c][t] == gt_gstate: 
                    if type_ is None:
                        self.growth_state_estimates[t] = 1 
                    else:
                        self.comparison_growth_state_estimates[type_][id_] = 1 
                else:
                    if type_ is None:
                        self.growth_state_estimates[t] = 0
                    else:
                        self.comparison_growth_state_estimates[type_][id_] = 0 

    def slam_update(self,t,slam,type_=None): 
        if type_ is None:
            self.slam_inst = slam 
            self.particles = slam.particles  
            best_particle_idx = np.argmax([x.weight for x in self.particles])
            self.best_traj_estimate[t,:] = self.particles[best_particle_idx].pose 
            best_landmark_estimates = self.particles[best_particle_idx].landmark
            lm_ids = [x.lm_id for x in best_landmark_estimates]
            for id_ in lm_ids:
                if id_ not in self.best_landmark_estimates.keys():
                    self.best_landmark_estimates[id_] = {}
                    self.best_landmark_estimates[id_]["mu"] = np.zeros((2,))
                    self.best_landmark_estimates[id_]["covar"] = np.zeros((2,2))
                idx = [i for i,x in enumerate(best_landmark_estimates) if x.lm_id][0]
                self.best_landmark_estimates[id_]["mu"] = best_landmark_estimates[idx].EKF.mu 
                self.best_landmark_estimates[id_]["covar"] = best_landmark_estimates[idx].EKF.Sigma 
        else:
            self.comparison_slam_instance[type_] = slam
            self.comparison_particles[type_] = slam.particles
            best_particle_idx = np.argmax([x.weight for x in slam.particles]) 
            if not type_ in self.comparison_best_traj_estimates.keys(): 
                self.comparison_best_traj_estimates[type_] = np.zeros_like(self.best_traj_estimate)
                self.comparison_best_landmark_estimates[type_] = {}     
            self.comparison_best_traj_estimates[type_][t,:] = slam.particles[best_particle_idx].pose 
            best_landmark_estimates = slam.particles[best_particle_idx].landmark
            lm_ids = [x.lm_id for x in best_landmark_estimates]
            for id_ in lm_ids:
                if id_ not in self.comparison_best_landmark_estimates[type_].keys():
                    self.comparison_best_landmark_estimates[type_][id_] = {}
                    self.comparison_best_landmark_estimates[type_][id_]["mu"] = np.zeros((2,))
                    self.comparison_best_landmark_estimates[type_][id_]["covar"] = np.zeros((2,2))
                idx = [i for i,x in enumerate(best_landmark_estimates) if x.lm_id][0]
                self.comparison_best_landmark_estimates[type_][id_]["mu"] = best_landmark_estimates[idx].EKF.mu 
                self.comparison_best_landmark_estimates[type_][id_]["covar"] = best_landmark_estimates[idx].EKF.Sigma 
    
    def update(self,t,clique,slam,type_=None):
        '''
        want to save best trajectory estimate, landmark location estimates
        want to save the posteriors
        '''
        self.update_clique_posteriors(t,clique,type_)
        self.slam_update(t,slam,type_)

