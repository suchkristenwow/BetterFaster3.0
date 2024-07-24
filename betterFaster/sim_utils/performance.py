import numpy as np 
import os 
import time 
import pickle 
import toml 
from .utils import get_data_association, get_sim_length, get_reinitted_id
import matplotlib.pyplot as plt 

def pickle_intermediate_results_background(performance_tracker, clique_sim, slam, t):
    # Assuming performance_tracker is an instance of your class
    performance_tracker.pickle_intermediate_results(clique_sim, slam, t)

def save_results_background(exp,parameters,performance_tracker,postProcessing_dirName): 
    background_process = write_results(exp,parameters,performance_tracker,postProcessing_dirName) 
    return background_process 

def performance_update_background(performance_tracker,t,clique_sim,slam,processing_time):
    performance_tracker.update(t,clique_sim,slam,processing_time) 

def write_results(experiment_no,exp_parameters,performance_tracker,out_dir):
    #results_dir = exp_parameters["results_dir"]
    sim_length = exp_parameters["sim_length"]
    #print("this is out_dir: ",out_dir)

    #save posterior estimate 
    if not os.path.exists(os.path.join(out_dir,"posteriors")):
        os.mkdir(os.path.join(out_dir,"posteriors"))
    
    posterior_dir = os.path.join(out_dir,"posteriors") 
    #print("this is posterior_dir: ",posterior_dir)
    print("writing {}".format(os.path.join(posterior_dir,"exp"+str(experiment_no)+".pickle")))
    with open(os.path.join(posterior_dir,"exp"+str(experiment_no)+".pickle"),"wb") as handle:
        pickle.dump(performance_tracker.posteriors,handle)

    #save best trajectory estimate 
    if not os.path.exists(os.path.join(out_dir,"trajectories")):
        os.mkdir(os.path.join(out_dir,"trajectories"))

    traj_dir = os.path.join(out_dir,"trajectories")
    print("writing {}".format(os.path.join(traj_dir,"exp"+str(experiment_no)+".csv")))
    np.savetxt(os.path.join(traj_dir,"exp"+str(experiment_no)+".csv"),performance_tracker.best_traj_estimate)

    #save best landmark localization estimates
    if not os.path.exists(os.path.join(out_dir,"lm_estimates")):
        os.mkdir(os.path.join(out_dir,"lm_estimates"))

    lm_dir = os.path.join(out_dir,"lm_estimates")
    print("writing {}".format(os.path.join(lm_dir,"exp"+str(experiment_no)+".pickle")))
    with open(os.path.join(lm_dir,"exp"+str(experiment_no)+".pickle"),"wb") as handle:
        pickle.dump(performance_tracker.best_landmark_estimates,handle)

    #save accuracy estimation 
    if not os.path.exists(os.path.join(out_dir,"accuracy")):
        os.mkdir(os.path.join(os.path.join(out_dir,"accuracy"))) 

    accuracy_dir = os.path.join(os.path.join(out_dir,"accuracy"))
    print("writing {}".format(os.path.join(accuracy_dir,"exp"+str(experiment_no)+".pickle")))
    with open(os.path.join(accuracy_dir,"exp"+str(experiment_no)+".pickle"),"wb") as handle:
        pickle.dump(performance_tracker.accuracy,handle) 

    #make slam error plots 
    save_slam_error_plots(experiment_no,sim_length,performance_tracker,out_dir)

def save_slam_error_plots(experiment_no,sim_length,performance_tracker,out_dir): 
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
    axs[1].plot(np.arange(last_t),performance_tracker.landmark_estimate_error_cache["mean"][:last_t],'b')
    #title_text = axs[1].set_title('Mean Landmark \nEstimate Error',rotation=90, verticalalignment='bottom', horizontalalignment='center')
    #title_text.set_position([-0.5, -0.5])
    axs[1].set_title('Mean Landmark \nEstimate Error',ha='left',va='center',x=-0.2,y=0.5,rotation=90)
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('MSE (m)')
    axs[1].set_xlim(0,last_t)
    print(max(performance_tracker.landmark_estimate_error_cache["mean"][:last_t])*1.05)
    if max(performance_tracker.landmark_estimate_error_cache["mean"][:last_t]) == 0:  
        axs[1].set_ylim(0,1)
    else:
        axs[1].set_ylim(0,max(performance_tracker.landmark_estimate_error_cache["mean"][:last_t])*1.05) 
    axs[1].grid(True)
    filename = "experiment"+str(experiment_no)+"_slam_err_plt.jpg" 
    if not os.path.exists(os.path.join(out_dir,"slam_err_plots")): 
        os.mkdir(os.path.join(out_dir,"slam_err_plots"))
    plt.savefig(os.path.join(out_dir,"slam_err_plots/" + filename))
    #plt.show(block=True)
    plt.close() 

class PerformanceTracker:
    def __init__(self,parameters,postProcessing_dir,comparison_type=None):
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

        if comparison_type is None: 
            results_dir = parameters["results_dir"]
        else: 
            results_dir = os.path.join(parameters["results_dir"],comparison_type) 
            postProcessing_dir = os.path.join(postProcessing_dir,comparison_type) 
            
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

            #experiment10data_association.csv
            data_association_path = os.path.join(results_dir,"data_association/experiment"+str(exp+1)+"data_association.csv")
            #print("data_associaiton_path: ",data_association_path)

            if parameters["isCarla"]:
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
                data_associations = np.genfromtxt(data_associations_path,delimiter=" ")
                if np.isnan(data_associations[:,0].any()):
                    data_associations = np.genfromtxt(data_associations_path,delimiter=" ")
                    print("this is data_associations: ",data_associations)
                    raise OSError 
            else:
                if data_associations is None:
                    raise OSError
            if np.isnan(data_associations).any():
                raise OSError
            else:
                self.all_data_associations[n] = data_associations

            for x in self.all_data_associations.keys():
                tmp = self.all_data_associations[x] 
                #print("tmp:",tmp)
                if np.isnan(tmp).any():
                    raise OSError
            exp_clique_ids = data_associations[:,0]
            self.gt_clique_persistence[n] = exp_clique_ids
        
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
        self.untuned_posteriors = None         
        self.growth_state_estimates = {}         
        self.processing_times = []

        #Error 
        self.ground_truth_trajectory = None 
        self.trajectory_estimate_error = np.zeros(((n_experiments*sim_length,2)))
        self.trajectory_estimate_error[:,0] = np.arange(n_experiments*sim_length)
        self.landmark_estimate_error_cache = {}
        self.landmark_estimate_error_cache["mean"] = np.zeros((n_experiments*sim_length,))
        self.landmark_estimate_error_cache["median"] =  np.zeros((n_experiments*sim_length,))
        self.landmark_estimate_error_cache["min"] =np.zeros((n_experiments*sim_length,))
        self.landmark_estimate_error_cache["max"] = np.zeros((n_experiments*sim_length,))

        self.ind_landmark_estimate_error_cache = {} 

        #Growth state errors 
        self.gstate_estimate_error_cache = {} 
        self.gstate_err_rate = np.zeros((n_experiments*self.sim_length,))

        self.isCarla = parameters['isCarla']
        self.postProcessing_dir = postProcessing_dir

    def compute_landmark_estimate_mse(self,t): 
        global_t = int(self.current_exp*self.sim_length + t) 
        #self.best_landmark_estimates[id_]["mu"] = best_landmark_estimates[idx].EKF.mu 
        #self.data_association[exp + 1]  = exp_data_association  
        lm_mses = []
        allZeros = []
        for id_ in self.best_landmark_estimates.keys(): 
            lm_mu = self.best_landmark_estimates[id_]["mu"]

            for i in self.data_association.keys(): 
                data_association = self.data_association[i]
                if id_ in data_association[:,0]: 
                    idx = np.where(data_association[:,0] == id_)
                    break 

            if not np.all(lm_mu == 0):
                gt_lm_loc = data_association[idx,1:]
                if len(gt_lm_loc) == 0 or gt_lm_loc.size == 0:
                    print("idx: ",idx) 
                    print("gt_lm_loc: ",gt_lm_loc)
                    raise OSError 
                lm_err = np.linalg.norm(gt_lm_loc - lm_mu)
                lm_mses.append(lm_err) 
                allZeros.append(True)
                if not id_ in self.ind_landmark_estimate_error_cache:
                    reinit_id = get_reinitted_id(self.all_data_associations,self.current_exp,id_)
                    if reinit_id in self.ind_landmark_estimate_error_cache:
                        id_ = reinit_id 
                    else:
                        self.ind_landmark_estimate_error_cache[id_] = np.zeros((self.sim_length*self.n_experiments,))
                self.ind_landmark_estimate_error_cache[id_][global_t] = lm_err  
            else:
                allZeros.append(False) 

        if np.all(allZeros == False): 
            print("WARNING: ALL LANDMARK ESTIMATES ARE ZERO")

        if np.isnan(np.mean(lm_mses)) or np.isinf(np.mean(lm_mses)): 
            #print("np.mean(lm_mses): ",np.mean(lm_mses))
            if len(lm_mses) > 0:
                print("np.mean(lm_mses): ",np.mean(lm_mses))
                print("lm_mses:",lm_mses)
                raise OSError 
        else:
            #print("global_t: ",type(global_t)) 
            self.landmark_estimate_error_cache["mean"][global_t] = np.mean(lm_mses)
            self.landmark_estimate_error_cache["median"][global_t] = np.median(lm_mses) 
            self.landmark_estimate_error_cache["min"][global_t] = min(lm_mses) 
            self.landmark_estimate_error_cache["max"][global_t] = max(lm_mses) 

        if not isinstance(np.mean(lm_mses),float): 
            print("lm_mses: ",lm_mses)
            raise OSError
        
        if np.mean(lm_mses) == 0 and len(lm_mses) > 0:
            print("lm_mses: ",lm_mses)
            raise OSError 
    
        if np.mod(t,10) == 0:
            #self.landmark_estimate_error_cache[global_t,1]  
            if global_t > 100: 
                t0 = global_t - 100 
                t1 = global_t - 90 
                mean_lm_err0 = np.mean(self.landmark_estimate_error_cache["mean"][t0:t1]) 
                tf = global_t - 10 
                mean_lm_err1 = np.mean(self.landmark_estimate_error_cache["mean"][tf:global_t]) 
                if mean_lm_err0 > mean_lm_err1: 
                    print("Dont worry ... Mean landmark estimation error is decreasing!") 
                    decrease_rate = (mean_lm_err0 - mean_lm_err1) / 100
                    print("decrease rate: ",np.round(decrease_rate,2))
                else:
                    increase_rate = (mean_lm_err1 - mean_lm_err0) / 100 
                    print("Mean landmark estimation error is locally increasing :/") 
                    print("increase rate: ",np.round(increase_rate,2)) 

            print("Mean Landmark Estimation Err: ",np.round(np.mean(lm_mses),3)) 

    def compute_trajectory_mse(self,t):
        global_t = int(self.current_exp*self.sim_length + t)
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
        self.clique_inst = clique 
        self.slam_inst = slam 
        self.acceptance_threshold = clique.acceptance_threshold
        self.rejection_threshold = clique.rejection_threshold
        self.posteriors = clique.posteriors
        self.particles = slam.particles 
        self.current_exp = experiment_no
        self.ground_truth_trajectory = gt_traj
        if parameters["isCarla"]:
            n_cliques_exp = self.gt_gstates[experiment_no + 1].shape[0]
        else:
            n_cliques_exp = self.gt_gstates[experiment_no].shape[0] 
        self.gstate_estimate_error_cache[experiment_no] = np.zeros((self.sim_length,n_cliques_exp))

    '''
    def init_new_comparison(self,type_,sim,slam): 
        self.comparison_sim_instance[type_] = sim 
        self.comparison_slam_instance[type_] = slam 
        if "untuned" in type_:
            self.untuned_posteriors = sim.posteriors
        self.comparison_particles[type_] = slam.particles 
    '''

    def update_clique_posteriors(self,t,clique):
        exp = self.current_exp
        self.clique_inst = clique 
        self.posteriors = clique.posteriors
        self.growth_state_estimates[exp] = clique.growth_state_estimates 
        '''
        if type_ is None:
            
        else:
            self.comparison_sim_instance[type_] = clique 
            self.comparison_particles[type_] = clique.posteriors 
            if type_ not in self.comparison_particles.keys():
                self.comparison_growth_state_estimates[type_] = {} 
            self.comparison_growth_state_estimates[type_][exp] = clique.growth_state_estimates 
        '''    
        #want to update accuracy of persistence
        for c in clique.posteriors.keys():
            posterior_c = clique.posteriors[c][t]
            #get_reinitted_id(all_data_associations,n,id_)
            reinit_id = get_reinitted_id(self.all_data_associations,self.current_exp,c)
            if reinit_id is not None: 
                id_ = reinit_id
            else:
                id_ = c 
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
         
        if (self.accuracy["true_positive"] + self.accuracy["false_negative"]) != 0 and (self.accuracy["true_negative"] + self.accuracy["false_positive"]) != 0: 
            sensitivity = self.accuracy["true_positive"] / (self.accuracy["true_positive"] + self.accuracy["false_negative"])
            specificity = self.accuracy["true_negative"] / (self.accuracy["true_negative"] + self.accuracy["false_positive"])
            balanced_accuracy = (sensitivity + specificity)/2
            if np.mod(t,10) == 0:
                print("Balanced Accuracy (posterior performance):",balanced_accuracy)
        else:
            '''
            print("true_positive: {}, true_negative: {}, false_positive: {} false_negative: {}".format(self.accuracy["true_positive"],self.accuracy["true_negative"],
                                                                                            self.accuracy["false_positive"],self.accuracy["false_negative"]))
            '''
            if np.mod(t,10) == 0:
                accuracy = (self.accuracy["true_positive"] + self.accuracy["true_negative"])/(self.accuracy["true_positive"] + self.accuracy["true_negative"] +\
                                                                                             self.accuracy["false_positive"] + self.accuracy["false_negative"])
                print("Accuracy (posterior performance): ",accuracy)
                
    def slam_update(self,t,slam): 
        #if type_ is None:
        self.slam_inst = slam 
        self.particles = slam.particles  
        best_particle_idx = np.argmax([x.weight for x in self.particles])
        self.best_traj_estimate[t,:] = self.particles[best_particle_idx].pose 
        best_landmark_estimates = self.particles[best_particle_idx].landmarks
        lm_ids = [x.lm_id for x in best_landmark_estimates]
        
        for id_ in lm_ids:
            if id_ not in self.best_landmark_estimates.keys():
                self.best_landmark_estimates[id_] = {}
                self.best_landmark_estimates[id_]["mu"] = np.zeros((2,))
                self.best_landmark_estimates[id_]["covar"] = np.zeros((2,2))
            idx = [i for i,x in enumerate(best_landmark_estimates) if x.lm_id == id_][0]
            '''
            self.best_landmark_estimates[id_]["mu"] = best_landmark_estimates[idx].EKF.mu 
            self.best_landmark_estimates[id_]["covar"] = best_landmark_estimates[idx].EKF.Sigma 
            '''
            self.best_landmark_estimates[id_]["mu"] = best_landmark_estimates[idx].mu 
            self.best_landmark_estimates[id_]["covar"] = best_landmark_estimates[idx].sigma 

    def compute_gstate_estimation_error(self,t,clique): 
        #print("self.current_exp: ",self.current_exp)

        if self.isCarla:
            exp_gstates = self.gt_gstates[self.current_exp + 1]
        else: 
            exp_gstates = self.gt_gstates[self.current_exp]

        if np.all(exp_gstates[:,1] == 0):
            print("exp_gstates:",exp_gstates)
            raise OSError 
        
        global_t = int(self.current_exp*self.sim_length + t) 
        err_t = 0
        current_gstate = []
        current_estimates = []        
        for c in clique.growth_state_estimates.keys():  
            #print("this is c: ",c)
            gt_gstate_c = None 
            orig_id = None 
            idx = None 
            
            if c not in exp_gstates[:,0]:
                orig_id = c #this is just debugging 
                id_ = get_reinitted_id(self.all_data_associations,self.current_exp,c,optional_exp=self.current_exp)
                if id_ not in exp_gstates[:,0]:
                    for i in self.all_data_associations.keys(): 
                        id_ = get_reinitted_id(self.all_data_associations,self.current_exp,c,optional_exp=i) 
                        if id_ in exp_gstates[:,0]: 
                            idx = np.where(exp_gstates[:,0] == id_)
                            gt_gstate_c = int(np.squeeze(exp_gstates[idx,1]))
                else:
                    idx = np.where(exp_gstates[:,0] == id_)  
                    idx = int(np.squeeze(idx))
                    gt_gstate_c = int(np.squeeze(exp_gstates[idx,1])) 
            else:
                idx = np.where(exp_gstates[:,0] == c)  
                gt_gstate_c = int(np.squeeze(exp_gstates[idx,1])) 

            if gt_gstate_c is None: 
                #print("this landmark is dead")
                if self.current_exp == min(self.all_data_associations.keys()):
                        print("This cant be dead")
                        print("self.gt_gstates: ",self.gt_gstates)
                        print("self.all_data_association: ",self.all_data_associations)
                        print("orig_id: ",orig_id) 
                        print("c: ",c)
                        print("exp_gstates: ",exp_gstates)
                        raise OSError 
                else:
                    gt_gstate_c = 0

            if gt_gstate_c is None:
                print("gt_gstate_c is None?")
                raise OSError 
            
            current_gstate.append(gt_gstate_c)

            estimated_gstate_c = clique.growth_state_estimates[c][global_t]

            current_estimates.append(estimated_gstate_c)

            if estimated_gstate_c != gt_gstate_c: 
                err_t += 1 
            else:
                if not idx is None:
                    self.gstate_estimate_error_cache[self.current_exp][idx] =  1 
        
        if np.all(current_gstate == 0):
            print("its not possible that these are all dead...")
            raise OSError 

        err_rate_t = err_t/len(clique.growth_state_estimates.keys())

        self.gstate_err_rate[global_t] = err_rate_t 

        if np.any(current_gstate is None): 
            print("current_gstate: ",current_gstate)
            raise OSError 
         
        if np.mod(t,10) == 0:
            '''
            if err_rate_t == 1:
                print("current_gstate: ",current_gstate)
                print("current_estimates: ",current_estimates)
                print("WARNING: All the GState Estimates are wrong!")
            '''
            print("Gstate Estimation Err Rate: ",err_rate_t)

    def update(self,t,clique,slam,processing_time):
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

        if np.mod(t,10) == 0:
            print("We are {} percent done!".format(np.round(percent_done,2)))

        if 10 < len(self.processing_times) < 1000:
            remaining_experiments = self.n_experiments - self.current_exp
            remaining_tsteps = self.sim_length - t  
            mean_secs_tstep = np.mean(self.processing_times)
            secs_remaining_for_this_exp = mean_secs_tstep * remaining_tsteps
            secs_remaining_for_all_exp = remaining_experiments * self.sim_length * mean_secs_tstep 
            total_secs_remaining = secs_remaining_for_all_exp + secs_remaining_for_this_exp
            total_min_remaining = total_secs_remaining / 60 
            if total_min_remaining > 100 and np.mod(t,10) == 0: 
                total_hours_remaining = total_min_remaining / 60  
                print("About {} hours remaining...".format(np.round(total_hours_remaining,3)))  
            elif np.mod(t,10) == 0:
                print("About {} minutes remaining...".format(np.round(total_min_remaining,1)))

            if 1000 <= len(self.processing_times): 
                #last_hundred_times = self.processing_times[-100:]
                q1= np.percentile(self.processing_times, 25)
                q3 = np.percentile(self.processing_times,75)  
                low_secs_remaining_for_this_exp = q1*remaining_tsteps 
                high_secs_remaining_for_this_exp = q3*remaining_tsteps  
                low_secs_remaining_for_all_exp = remaining_experiments*self.sim_length*q1
                high_secs_remaining_for_all_exp = remaining_experiments*self.sim_length*q3 
                low_total_secs_remaining = low_secs_remaining_for_this_exp + low_secs_remaining_for_all_exp 
                high_total_secs_remaining = high_secs_remaining_for_this_exp + high_secs_remaining_for_all_exp 
                low_mins_remaining = low_total_secs_remaining/60 
                high_mins_remaining = high_total_secs_remaining/60 
                if np.mod(t,10) == 0:
                    if low_mins_remaining > 100:
                        low_hours_remaining = low_mins_remaining/60 
                        high_hours_remaining = high_mins_remaining/60  
                        print("Between {} and {} hours remaining!!!".format(np.round(low_hours_remaining,2),np.round(high_hours_remaining,2))) 
                    else: 
                        print("Between {} and {} minutes remaining!!!".format(np.round(low_mins_remaining,1),np.round(high_mins_remaining,1))) 
                '''
                if low_mins_remaining > 100 and np.mod(t,10) == 0: 
                    low_hours_remaining = low_mins_remaining/60 
                    high_hours_remaining = high_mins_remaining/60  
                    print("Between {} and {} hours remaining!!!".format(np.round(low_hours_remaining,2),np.round(high_hours_remaining,2)))
                elif np.mod(t,10) == 0:
                    print("Between {} and {} minutes remaining!!!".format(np.round(low_mins_remaining,1),np.round(high_mins_remaining,1)))
                '''

        self.update_clique_posteriors(t,clique)
        self.slam_update(t,slam)
        self.compute_landmark_estimate_mse(t) 
        self.compute_trajectory_mse(t) 

        '''
        if 0 not in self.gt_gstates.keys():
            print("gt_gstates:",self.gt_gstates[self.current_exp +1])
        else: 
            print("gt_gstates:",self.gt_gstates[self.current_exp]) 
        if 3605 not in self.gt_gstates[self.current_exp + 1] and self.current_exp ==2:
                raise OSError 
        else:
            print("gt_gstates:",self.gt_gstates[self.current_exp]) 
            if 3605 not in self.gt_gstates[self.current_exp] and self.current_exp == 2:
                raise OSError 
        ''' 

        self.compute_gstate_estimation_error(t,clique)
    
    def pickle_intermediate_results(self,clique,slam,t): 
        if not os.path.exists(os.path.join(self.postProcessing_dir,"int_results")): 
            os.mkdir(os.path.join(self.postProcessing_dir,"int_results"))

        intermediate_pickle_name = "exp"+str(self.current_exp)+"_int_clique_states"+str(t)+".pickle"
        int_clique_states = {}
        for c in clique.tracked_cliques.keys():
            #JPF stuff
            clique_likelihood_c = clique.tracked_cliques[c]._clique_likelihood #this is a float
            log_clique_evidence_c = clique.tracked_cliques[c]._log_clique_evidence #this is also a float 
            last_observation_time = clique.tracked_cliques[c]._last_observation_time #this is an array of ints (len = n_feats)
            log_likelihood = clique.tracked_cliques[c]._log_likelihood #this is also an array of ints (len = n_feats)
            log_clique_lower_evidence_sum = clique.tracked_cliques[c]._log_clique_lower_evidence_sum 
            init_tstep = clique.tracked_cliques[c].init_tstep 
            #print("this is init_tstep: ",init_tstep)
            #pickling!
            int_clique_states[c] = {} 
            int_clique_states[c]["clique_likelihood"] = clique_likelihood_c
            int_clique_states[c]["log_clique_evidence"] = log_clique_evidence_c
            int_clique_states[c]["last_observation_time"] = last_observation_time
            int_clique_states[c]["log_likelihood"] = log_likelihood
            int_clique_states[c]["log_clique_lower_evidence_sum"] = log_clique_lower_evidence_sum
            int_clique_states[c]["init_tstep"] = init_tstep 
            #print("int_clique_states[c].keys(): ",int_clique_states[c].keys())

        #other 
        growth_state_estimates = clique.growth_state_estimates  
        cone_feature_description_cache = clique.cone_feature_description_cache 
        tree_feature_description_cache = clique.tree_feature_description_cache 
        int_clique_states["growth_state_estimates"] = growth_state_estimates
        int_clique_states["cone_feature_description_cache"] = cone_feature_description_cache
        int_clique_states["tree_feature_description_cache"] = tree_feature_description_cache
        int_clique_states["posteriors"] = clique.posteriors  
        int_clique_states["observation_cache"] = clique.observation_cache

        if t > 100: 
            for id_ in clique.posteriors.keys(): 
                posteriors_id = clique.posteriors[id_] 
                last_posteriors = posteriors_id[t-100:t] 
                if np.all(last_posteriors) == 0: 
                    print() 
                    print("WARNING the last 100 posteriors for this clique: {} are zero".format(id_))
                    print() 
                    
        #print("int_clique_states.keys(): ",int_clique_states.keys()) 
        for c in int_clique_states.keys():
            if isinstance(c,int):
                if "init_tstep" not in int_clique_states[c].keys():
                    print("int_clique_states[c].keys(): ",int_clique_states[c].keys())
                    raise OSError 

        with open(os.path.join(self.postProcessing_dir,"int_results/" + intermediate_pickle_name),"wb") as handle:
            #print("writing: ",os.path.join(self.postProcessing_dir,"int_results/" + intermediate_pickle_name))
            pickle.dump(int_clique_states,handle)
        
        intermediate_pickle_name = "exp"+str(self.current_exp)+"_int_slam_results"+str(t)+".pickle" 
        slam_pickle = {} 
        for k in range(len(slam.particles)): 
            slam_pickle[k] = {}
            #each particle has properties: pose, weight, and landmark
            particle_k = slam.particles[k]
            part_weight = particle_k.weight 
            part_pose = particle_k.pose 
            slam_pickle[k]["weight"] = part_weight 
            slam_pickle[k]["pose"] = part_pose
            part_landmark = particle_k.landmarks 
            slam_pickle[k]["landmarks"] = {}
            #each landmark has properties: EKF, isobserved, and lm_id 
            for lm in part_landmark: 
                lm_obsd = lm.isobserved 
                lm_id = lm.lm_id 
                slam_pickle[k]["landmarks"][lm_id] = {} 
                slam_pickle[k]["landmarks"][lm_id]["isobserved"] = lm_obsd 
                #EKF_p = lm.weight 
                slam_pickle[k]["landmarks"][lm_id]["mu"] = lm.mu 
                slam_pickle[k]["landmarks"][lm_id]["sigma"] = lm.sigma 
                #slam_pickle[k]["landmarks"][lm_id]["p"] = EKF_p

        with open(os.path.join(self.postProcessing_dir,"int_results/" + intermediate_pickle_name),"wb") as handle: 
            pickle.dump(slam_pickle,handle)  

        #pickle performance stuff 
        int_perf_pickle = {} 
        int_perf_pickle["accuracy"] = {} 
        int_perf_pickle["accuracy"]["true_positive"] = self.accuracy["true_positive"]
        int_perf_pickle["accuracy"]["true_negative"] = self.accuracy["true_negative"]
        int_perf_pickle["accuracy"]["false_positive"] = self.accuracy["false_positive"]
        int_perf_pickle["accuracy"]["false_negative"] = self.accuracy["false_negative"]
        int_perf_pickle["growth_state_estimates"]  = self.growth_state_estimates
        int_perf_pickle["trajectory_estimate_error"] = self.trajectory_estimate_error
        int_perf_pickle["landmark_estimate_error_cache"] = self.landmark_estimate_error_cache 
        int_perf_pickle["ind_landmark_estimate_error_cache"] = self.ind_landmark_estimate_error_cache
        int_perf_pickle["gstate_error_cache"] = self.gstate_estimate_error_cache
        int_perf_pickle["gstate_err_rate"] = self.gstate_err_rate
        int_perf_pickle["processing_time"] = self.processing_times

        intermediate_pickle_name = "exp"+str(self.current_exp)+"_int_performance_results"+str(t)+".pickle" 
        with open(os.path.join(self.postProcessing_dir,"int_results/" + intermediate_pickle_name),"wb") as handle: 
            pickle.dump(int_perf_pickle,handle)  

    def load_prev_results(self,pickle_path): 
        with open(pickle_path,"rb") as handle:
            prev_results = pickle.load(handle)
        self.accuracy["true_positive"] = prev_results["accuracy"]["true_positive"]
        self.accuracy["true_negative"] = prev_results["accuracy"]["true_negative"]
        self.accuracy["false_positive"] = prev_results["accuracy"]["false_positive"]
        self.accuracy["false_negative"] = prev_results["accuracy"]["false_negative"]
        self.growth_state_estimates = prev_results["growth_state_estimates"]
        self.trajectory_estimate_error = prev_results["trajectory_estimate_error"] 
        self.landmark_estimate_error_cache = prev_results["landmark_estimate_error_cache"] 
        self.ind_landmark_estimate_error_cache = prev_results["ind_landmark_estimate_error_cache"]
        self.gstate_estimate_error_cache = prev_results["gstate_error_cache"] 
        self.gstate_err_rate = prev_results["gstate_err_rate"]
        if "processing_time" in prev_results.keys():
            self.processing_times = prev_results["processing_time"]
