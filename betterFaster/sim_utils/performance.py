import numpy as np 
import os 
import time 
import pickle 
import toml 
from .utils import get_data_association, get_sim_length, get_reinitted_id
import matplotlib.pyplot as plt 

def write_results(experiment_no,exp_parameters,performance_tracker):
    results_dir = exp_parameters["results_dir"]
    sim_length = exp_parameters["sim_length"]

    if not os.path.exists(os.path.join(results_dir,"postProcessingResults")):
        os.mkdir(os.path.join(results_dir,"postProcessingResults"))

    postProcessingResults_dir = os.path.join(results_dir,"postProcessingResults")

    #save posterior estimate 
    if not os.path.exists(os.path.join(results_dir,"postProcessingResults/posteriors")):
        os.mkdir(os.path.join(results_dir,"postProcessingResults/posteriors"))
    
    posterior_dir = os.path.join(results_dir,"postProcessingResults/posteriors") 
    print("writing {}".format(os.path.join(posterior_dir,"exp"+str(experiment_no)+".pickle")))
    with open(os.path.join(posterior_dir,"exp"+str(experiment_no)+".pickle"),"wb") as handle:
        pickle.dump(performance_tracker.posteriors,handle)

    #save best trajectory estimate 
    if not os.path.exists(os.path.join(results_dir,"postProcessingResults/trajectories")):
        os.mkdir(os.path.join(results_dir,"postProcessingResults/trajectories"))

    traj_dir = os.path.join(postProcessingResults_dir,"trajectories")
    print("writing {}".format(os.path.join(traj_dir,"exp"+str(experiment_no)+".csv")))
    np.savetxt(os.path.join(traj_dir,"exp"+str(experiment_no)+".csv"),performance_tracker.best_traj_estimate)

    #save best landmark localization estimates
    if not os.path.exists(os.path.join(results_dir,"postProcessingResults/lm_estimates")):
        os.mkdir(os.path.join(results_dir,"postProcessingResults/lm_estimates"))

    lm_dir = os.path.join(postProcessingResults_dir,"lm_estimates")
    print("writing {}".format(os.path.join(lm_dir,"exp"+str(experiment_no)+".pickle")))
    with open(os.path.join(lm_dir,"exp"+str(experiment_no)+".pickle"),"wb") as handle:
        pickle.dump(performance_tracker.best_landmark_estimates,handle)

    #save accuracy estimation 
    if not os.path.exists(os.path.join(results_dir,"postProcessingResults/accuracy")):
        os.mkdir(os.path.join(results_dir,"postProcessingResults/accuracy"))

    accuracy_dir = os.path.join(postProcessingResults_dir,"accuracy")
    print("writing {}".format(os.path.join(accuracy_dir,"exp"+str(experiment_no)+".pickle")))
    with open(os.path.join(accuracy_dir,"exp"+str(experiment_no)+".pickle"),"wb") as handle:
        pickle.dump(performance_tracker.accuracy,handle) 

    #make slam error plots 
    save_slam_error_plots(results_dir,experiment_no,sim_length,performance_tracker)

def save_slam_error_plots(results_dir,experiment_no,sim_length,performance_tracker): 
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))  # 2 Rows, 1 Column 
    fig.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.05)
    #trajectory error 
    last_t = (experiment_no + 1)*sim_length 
    axs[0].plot(performance_tracker.trajectory_estimate_error[:last_t,0],performance_tracker.trajectory_estimate_error[:last_t,1],'r')
    #title_text = axs[0].set_title('Trajectory Error',rotation=90, verticalalignment='bottom', horizontalalignment='center')
    #title_text.set_position([-0.5, -0.5])
    axs[0].set_title('Trajectory Error',ha='left',va='center',x=-0.2,y=0.5,rotation=90)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('MSE (m)')
    axs[0].set_xlim(0,last_t) 
    axs[0].set_ylim(0,max(performance_tracker.trajectory_estimate_error[:last_t,1])*1.05)
    axs[0].grid(True)
    #mean landmark estimation error 
    axs[1].plot(performance_tracker.landmark_estimate_error_cache[:last_t,0],performance_tracker.landmark_estimate_error_cache[:last_t,1],'b')
    #title_text = axs[1].set_title('Mean Landmark \nEstimate Error',rotation=90, verticalalignment='bottom', horizontalalignment='center')
    #title_text.set_position([-0.5, -0.5])
    axs[1].set_title('Mean Landmark \nEstimate Error',ha='left',va='center',x=-0.2,y=0.5,rotation=90)
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('MSE (m)')
    axs[1].set_xlim(0,last_t)
    axs[1].set_ylim(0,10)
    axs[1].grid(True)
    #plt.tight_layout() 
    filename = "experiment"+str(experiment_no)+"_slam_err_plt.jpg" 
    if not os.path.exists(os.path.join(results_dir,"slam_err_plots")): 
        os.mkdir(os.path.join(results_dir,"slam_err_plots"))
    plt.savefig(os.path.join(results_dir,"slam_err_plots/" + filename))
    #plt.show(block=True)
    plt.close() 

class PerformanceTracker:
    def __init__(self,parameters):
        '''
        if data_associations is None:
            all_results_dir = "/media/arpg/easystore1/BetterFaster/kitti_carla_simulator/"
            if not os.path.exists(os.path.join(all_results_dir,"exp_results")): 
                all_results_dir = "/media/arpg/easystore/BetterFaster/kitti_carla_simulator/"
                if not os.path.exists(os.path.join(all_results_dir,"exp_results")):
                    raise OSError
            results_dir = os.path.join(all_results_dir,"exp_results")
        else:
            results_dir = "sim_utils/fake_data"
        '''
        n_experiments = parameters["experiments"]
        results_dir = parameters["results_dir"]
        data_associations = get_data_association(n_experiments,results_dir) 

        with open(os.path.join(results_dir,"gt_gstates.pickle"),"rb") as handle: 
            self.gt_gstates = pickle.load(handle)

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
        #print("in performance... getting data associations....")
        for n in range(1,n_experiments+1):
            #print("this is exp no:{}...".format(n))
            if not self.results_dir is None:
                #dir_n = os.path.join(self.results_dir,"exp"+str(n)+"_results")
                data_associations_path = os.path.join(results_dir,"data_association/experiment"+str(n)+"data_association.csv")
                if not os.path.exists(data_associations_path):
                    data_associations_path = os.path.join(results_dir,"data_associations/exp"+str(n-1)+"_data_association.csv")
                    if not os.path.exists(data_associations_path):
                        print("data_associations_path:",data_associations_path)
                        raise OSError
                #print("data_associations_path: ",data_associations_path)
                data_associations = np.genfromtxt(data_associations_path)
                if np.isnan(data_associations[:,0].any()):
                    data_associations = np.genfromtxt(data_associations_path,delimiter=" ")
                    #print("this is data_associations: ",data_associations)
            else:
                if data_associations is None:
                    raise OSError
            if np.isnan(data_associations).any():
                raise OSError
            else:
                self.all_data_associations[n] = data_associations
                #print("assigning data_associations to self.all_data_associations... this is n:",n)
            #sanity check
            #print("checking the data associations....")
            for x in self.all_data_associations.keys():
                #print("checking what we've set... this is n:",x)
                tmp = self.all_data_associations[x] 
                #print("tmp:",tmp)
                if np.isnan(tmp).any():
                    raise OSError
            exp_clique_ids = data_associations[:,0]
            #print("exp_clique_ids: ",exp_clique_ids)
            if n > 1 and self.results_dir is not None:
                for i,id_ in enumerate(exp_clique_ids):
                    #get_reinitted_id(all_data_associations,n,id_)
                    orig_id = get_reinitted_id(self.all_data_associations,n,id_) #this should be the original clique id
                    #print("this is orig_id: ",orig_id)
                    if not orig_id is None:
                        exp_clique_ids[i] = orig_id
            self.gt_clique_persistence[n] = exp_clique_ids
            #print("this is all_data_associations: ",self.all_data_associations[n]) 
        with open(os.path.join(self.results_dir,"gt_gstates.pickle"),"rb") as handle:
            self.gt_gstates = pickle.load(handle)
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
        self.growth_state_estimates = {} 
        self.comparison_growth_state_estimates = {} 
        self.processing_times = []

        #Error 
        self.ground_truth_trajectory = None 
        self.trajectory_estimate_error = np.zeros(((n_experiments*sim_length,2)))
        self.trajectory_estimate_error[:,0] = np.arange(n_experiments*sim_length)
        self.landmark_estimate_error_cache = np.zeros((n_experiments*sim_length,2))
        self.landmark_estimate_error_cache[:,0] = np.arange(n_experiments*sim_length)
        #Growth state errors 
        self.gstate_estimate_error_cache = {} 
        self.gstate_err_rate = np.zeros((n_experiments*self.sim_length,))

    def compute_landmark_estimate_mse(self,t): 
        global_t = self.current_exp*self.sim_length + t 
        #self.best_landmark_estimates[id_]["mu"] = best_landmark_estimates[idx].EKF.mu 
        #self.data_association[exp + 1]  = exp_data_association  
        lm_mses = []
        
        for id_ in self.best_landmark_estimates.keys(): 
            lm_mu = self.best_landmark_estimates[id_]["mu"]
            idx = np.where(self.data_association[self.current_exp + 1][:,0] == id_)
            if not np.all(lm_mu == 0):
                gt_lm_loc = self.data_association[self.current_exp + 1][idx,1:]
                lm_err = np.linalg.norm(gt_lm_loc - lm_mu)
                lm_mses.append(lm_err) 

        if np.isnan(np.mean(lm_mses)) or np.isinf(np.mean(lm_mses)): 
            if len(lm_mses) == 0:
                self.landmark_estimate_error_cache[global_t,1] = 0 
            else:
                print("np.mean(lm_mses): ",np.mean(lm_mses))
                print("lm_mses:",lm_mses)
                raise OSError 
        else:
            self.landmark_estimate_error_cache[global_t,1] = np.mean(lm_mses)

    def compute_trajectory_mse(self,t):
        global_t = self.current_exp*self.sim_length + t
        pose_t = self.best_traj_estimate[t,:2]
        gt_pose = self.ground_truth_trajectory[t,:2] 
        err_ = np.linalg.norm(pose_t - gt_pose) 
        if err_ > 100: 
            print("err_:",err_)
            print("pose_t: {}".format(pose_t))
            print("gt_pose: ",gt_pose)
            raise OSError 
        if np.isnan(err_) or np.isinf(err_): 
            print("pose_t: ",pose_t)
            print("gt_pose:",gt_pose)
            print("err_: ",err_)
            raise OSError 
        self.trajectory_estimate_error[global_t,1] = err_

    def init_new_experiment(self,experiment_no,clique,slam,gt_traj,parameters):
        compare_betterTogether = parameters["comparison_bools"]["compare_betterTogether"]
        compare_vanilla = parameters["comparison_bools"]["compare_vanilla"]
        compare_multiMap = parameters["comparison_bools"]["compare_multiMap"]
        #compare_betterTogether,compare_multiMap,compare_vanilla
        self.clique_inst = clique 
        self.slam_inst = slam 
        self.acceptance_threshold = clique.acceptance_threshold
        self.rejection_threshold = clique.rejection_threshold
        self.posteriors = clique.posteriors
        self.particles = slam.particles 
        self.current_exp = experiment_no
        self.ground_truth_trajectory = gt_traj
        n_cliques_exp = self.gt_gstates[experiment_no].shape[0]
        self.gstate_estimate_error_cache[experiment_no] = np.zeros((self.sim_length,n_cliques_exp))

        #self.clique_sim = clique 
        #self.BF_SLAM = slam 

    def init_new_comparison(self,type_,sim,slam): 
        self.comparison_sim_instance[type_] = sim 
        self.comparison_slam_instance[type_] = slam 
        if "untuned" in type_:
            self.untuned_posteriors = sim.posteriors
        self.comparison_particles[type_] = slam.particles 

    def update_clique_posteriors(self,t,clique,type_=None):
        exp = self.current_exp
        if type_ is None:
            self.clique_inst = clique 
            self.posteriors = clique.posteriors
            self.growth_state_estimates[exp] = clique.growth_state_estimates 
        else:
            self.comparison_sim_instance[type_] = clique 
            self.comparison_particles[type_] = clique.posteriors 
            if type_ not in self.comparison_particles.keys():
                self.comparison_growth_state_estimates[type_] = {} 
            self.comparison_growth_state_estimates[type_][exp] = clique.growth_state_estimates 
            
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
                if id_ in self.gt_clique_persistence[self.current_exp + 1]: 
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
                    '''
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
                    '''
                    #reinit_id = get_reinitted_id(self.all_data_associations,self.current_exp,c) 
                    if c in self.gt_gstates[self.current_exp][:,0] or get_reinitted_id(self.all_data_associations,self.current_exp,c) in self.gt_gstates[self.current_exp][:,0]: 
                        if c in self.gt_gstates[self.current_exp][:,0]: 
                            idx = np.where(self.gt_gstates[self.current_exp][:,0] == c)
                        else: 
                            reinit_id = get_reinitted_id(self.all_data_associations,self.current_exp,c) 
                            idx = np.where(self.gt_gstates[self.current_exp][:,0] == reinit_id)
                        gt_gstate = self.gt_gstates[self.current_exp][idx,1]
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
                idx = [i for i,x in enumerate(best_landmark_estimates) if x.lm_id == id_][0]
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
                idx = [i for i,x in enumerate(best_landmark_estimates) if x.lm_id == id_][0]
                self.comparison_best_landmark_estimates[type_][id_]["mu"] = best_landmark_estimates[idx].EKF.mu 
                self.comparison_best_landmark_estimates[type_][id_]["covar"] = best_landmark_estimates[idx].EKF.Sigma 

    def compute_gstate_estimation_error(self,t,clique): 
        #self.gt_gstates
        exp_gstates = self.gt_gstates[self.current_exp] 
        global_t = self.current_exp*self.sim_length + t 
        err_t = 0
        for c in clique.growth_state_estimates.keys():  
            print("this is c: ",c)
            gt_gstate_c = None 
            idx = np.where(exp_gstates[:,0] == c) 
            print("this is idx: ",idx)
            if c not in exp_gstates[:,0]:
                reinit_id = get_reinitted_id(self.all_data_associations,self.current_exp,c) 
                print("this is reinit_id: ",reinit_id)
                if reinit_id not in exp_gstates[:,0]:
                    gt_gstate_c = 0 
                else:
                    idx = np.where(exp_gstates[:,0] == reinit_id)

            print("exp_gstates: ",exp_gstates)
            print("idx: ",idx)
            print("exp_gstates[idx,1]: ",exp_gstates[idx,1])
            print("type of exp_gstates[idx,1]: ",type(exp_gstates[idx,1]))
            print("shape: ",exp_gstates[idx,1].shape)
            print("np.squeeze(exp_gstates[idx,1]): ",np.squeeze(exp_gstates[idx,1]))
            if not gt_gstate_c == 0:
                gt_gstate_c = int(np.squeeze(exp_gstates[idx,1]))

            estimated_gstate_c = clique.growth_state_estimates[c][global_t] 
            print("estimated_gstate_c: ",estimated_gstate_c)
            print("gt_gstate_c: ",gt_gstate_c)

            if estimated_gstate_c != gt_gstate_c: 
                err_t += 1 
            else:
                if not idx is None:
                    self.gstate_estimate_error_cache[self.current_exp][idx] =  1 

        err_rate_t = err_t/len(clique.growth_state_estimates.keys())

        self.gstate_err_rate[global_t] = err_rate_t 

    def update(self,t,clique,slam,processing_time,type_=None):
        '''
        want to save best trajectory estimate, landmark location estimates
        want to save the posteriors
        '''
        if clique is None:
            raise OSError 
        
        if not processing_time is None:
            self.processing_times.append(processing_time) 
        total_tsteps = self.sim_length * self.n_experiments 
        completed_tsteps =  (self.current_exp)*self.sim_length + t 
        percent_done = 100*completed_tsteps/total_tsteps  
        print("We are {} percent done!".format(np.round(percent_done,2)))
        if len(self.processing_times) > 10:
            remaining_experiments = self.n_experiments - self.current_exp
            remaining_tsteps = self.sim_length - t  
            mean_secs_tstep = np.mean(self.processing_times)
            secs_remaining_for_this_exp = mean_secs_tstep * remaining_tsteps
            secs_remaining_for_all_exp = remaining_tsteps * self.sim_length * mean_secs_tstep 
            total_secs_remaining = secs_remaining_for_all_exp + secs_remaining_for_this_exp
            total_min_remaining = total_secs_remaining / 60 
            if total_min_remaining > 100: 
                total_hours_remaining = total_min_remaining / 60  
                print("About {} hours remaining...".format(np.round(total_hours_remaining,3))) 
            else:
                print("About {} minutes remaining...".format(np.round(total_min_remaining,1)))
        #exp,t,clique
        #self.update_clique_posteriors(t,clique,type_)
        self.update_clique_posteriors(t,clique)
        self.slam_update(t,slam,type_)

        self.compute_landmark_estimate_mse(t) 
        self.compute_trajectory_mse(t)

        self.compute_gstate_estimation_error(t,clique)