import numpy as np 
import pickle 
import argparse 
import os 
import time 
import toml 
import multiprocessing
from betterFaster import * 

class RunComparison: 
    def __init__(self,args,performance_tracker,data_association_rate,comparison_type): 
        self.args = args 
        self.comparison_type = comparison_type 
        with open(self.args.config,"r") as f:
            self.parameters = toml.load(f)
        self.performance_tracker = performance_tracker
        self.already_loaded_prev_results = False 
        self.args.postProcessing_dirName = os.path.join(self.args.postProcessing_dirName,comparison_type + "_" + str(data_association_rate)) 
        self.data_association_rate = data_association_rate  

    def run(self,exp): 
        if exp < self.args.start_experiment:
            print("we are skipping this experiment: {} as per the command line arguments...".format(exp))
            return exp 

        #1. Load in Data
        results_dir = os.path.join(self.parameters["results_dir"]) 

        sim_length = self.parameters["sim_length"]

        if self.args.load_prev_exp_results: 
            avilable_start_exp, avilable_start_tstep = get_start_time_exp(self.args.postProcessing_dirName,self.args.start_experiment,self.args.start_tstep,sim_length)
            if not self.already_loaded_prev_results: 
                if exp != avilable_start_exp or avilable_start_tstep != self.args.start_tstep:
                    print("Sorry we can't run that experiment. Going to start at exp:{},timestep:{} instead".format(avilable_start_exp,avilable_start_tstep))
                    input("Press enter to continue ...")
                '''
                if avilable_start_tstep != self.args.start_tstep:
                    print("Sorry we don't have results starting at that timestep. Going to start at exp:{}, timestep:{} instead".format(avilable_start_exp,avilable_start_tstep)) 
                    input("Press enter to continue ...")
                '''
                exp = avilable_start_exp; self.args.start_experiment = avilable_start_exp; self.args.start_tstep = avilable_start_tstep
                self.already_loaded_prev_results = True 
                #postProcessing_dirName
                if os.path.exists(os.path.join(self.args.postProcessing_dirName,"int_results/exp"+str(exp)+"_int_performance_results"+str(avilable_start_tstep)+".pickle")): 
                    performance_tracker.load_prev_results(os.path.join(self.args.postProcessing_dirName,"int_results/exp"+str(exp)+"_int_performance_results"+str(avilable_start_tstep)+".pickle"))
                print("loaded in exp results!")
        '''
        else: 
            if exp == 0:
                purge = input("Would you like to purge old experimental results? Type Y if you want to delete the previous intermediate results and N otherwise.")            
                if purge == "Y": 
                    int_result_dir = os.path.join(self.args.postProcessing_dirName,"int_results")
                    os.rename(int_result_dir,os.path.join(self.args.postProcessing_dirName,"int_results-old"))
        '''
        if self.comparison_type in ["betterTogether","multiMap"]: 
            if self.parameters['isCarla']:
                #print("this is exp: ",exp)
                '''
                if os.path.exists(os.path.join(results_dir,"observation_pickles/experiment"+str(exp)+"all_clique_feats.pickle")): 
                    clique_feats_path = os.path.join(results_dir,"observation_pickles/experiment"+str(exp)+"all_clique_feats.pickle")
                else: 
                '''
                clique_feats_path = os.path.join(results_dir,"observation_pickles/experiment"+str(exp + 1)+"all_clique_feats.pickle")
                #print("this is clique feats path: ",clique_feats_path)
            else:
                clique_feats_path = os.path.join(results_dir,"exp"+str(exp)+"all_clique_feats.pickle")

            with open(clique_feats_path,"rb") as handle:
                all_clique_feats = pickle.load(handle)

        if self.parameters["isCarla"]:
            #obsd_clique_path = os.path.join(results_dir,"observation_pickles/experiment"+str(exp + 1)+"observed_cliques.pickle")
            print("this is exp:",exp)
            print(os.path.join(results_dir,"reformed_carla_observations/exp"+str(exp)+"reformed_carla_observations.pickle")) 
            if os.path.exists(os.path.join(results_dir,"reformed_carla_observations/exp"+str(exp)+"reformed_carla_observations.pickle")): 
                with open(os.path.join(results_dir,"reformed_carla_observations/exp"+str(exp)+"reformed_carla_observations.pickle"),"rb") as handle:
                    exp_observations = pickle.load(handle)
            else: 
                input("The processed experiment observations path does not exist")
        else:
            obsd_clique_path = os.path.join(results_dir,"observation_pickles/exp"+str(exp + 1)+"observed_cliques.pickle")
            with open(obsd_clique_path,"rb") as handle: 
                exp_observations = pickle.load(handle)

        #DEBUG# 
        observed_clique_ids = []
        for t in exp_observations.keys(): 
            if np.mod(t,10): 
                observations_t = exp_observations[t]
                observed_cliques_t = [x["clique_id"] for x in observations_t] 
                observed_clique_ids.extend([x for x in observed_cliques_t if x not in observed_clique_ids]) 
        observed_clique_ids = np.unique(observed_clique_ids)
        #print("all_observed_ids: ",np.unique(observed_clique_ids))
        
        if self.comparison_type in ["betterTogether","multiMap"]:   
            for x in observed_clique_ids: 
                if x not in all_clique_feats.keys():
                    reinitted_id = get_reinitted_id(performance_tracker.all_data_associations,exp,x,optional_exp=exp)
                    orig_id = get_reinitted_id(performance_tracker.all_data_associations,exp,x)
                    if reinitted_id not in all_clique_feats.keys() and orig_id not in all_clique_feats.keys():
                        print("x: {}, orig_id: {}, reinitted_id: {}, all_clique_feats.keys(): {}".format(x,orig_id,reinitted_id,all_clique_feats.keys()))
                        raise OSError 
        
        assert len(exp_observations) > 0, "No observations for this experiment?"

        #Ground truth trajectories of the car    
        if not self.parameters["isCarla"]:
            gt_car_traj = np.genfromtxt(os.path.join(self.parameters["results_dir"],"fake_gt_traj.csv"),delimiter=",")
        else: 
            orig_gt_car_traj = np.genfromtxt(os.path.join(self.parameters["results_dir"],"gt_car_poses/experiment"+str(exp + 1)+"_gt_car_pose.csv"),delimiter=",")
            gt_car_traj = np.zeros((orig_gt_car_traj.shape[0],3))
            gt_car_traj[:,0] = orig_gt_car_traj[:,0]
            gt_car_traj[:,1] = orig_gt_car_traj[:,1]
            gt_car_traj[:,2] = orig_gt_car_traj[:,5]

        localization_covariance_params = self.parameters["slam"]["localization_covariance_params"]
        var_x = localization_covariance_params[0]; var_y = localization_covariance_params[1]; var_theta = localization_covariance_params[2]
        localization_covariance = np.diag([var_x,var_y,var_theta])
        
        '''
        if self.parameters["isCarla"]: 
            sim_utils = simUtils(exp,performance_tracker.all_data_associations,self.parameters)
        '''

        #2. Instantiate Simulator Classes 
        print("instantiating classes")
        if self.args.start_experiment > 0 and self.args.load_prev_exp_results:
            if self.comparison_type == "betterTogether": 
                clique_sim = betterTogether_simulator(self.parameters,all_clique_features=all_clique_feats,current_exp=exp,sim_length=sim_length,
                                            data_association=performance_tracker.all_data_associations,exp_observations=exp_observations,exp_obs_ids=observed_clique_ids,
                                            feature_detection=self.args.enable_feature_detection,verbosity=self.args.verbose,start_time=(self.args.start_experiment,self.args.start_tstep))
            '''
            clique_sim = clique_simulator(self.parameters,all_clique_features=all_clique_feats,current_exp=exp,sim_length=sim_length,
                                            data_association=performance_tracker.all_data_associations,exp_observations=exp_observations,exp_obs_ids=observed_clique_ids,
                                            feature_detection=self.args.enable_feature_detection,verbosity=self.args.verbose,start_time=(self.args.start_experiment,self.args.start_tstep))
            '''
            if self.comparison_type == "multiMap": 
                multiMap_sim = multiMap_simulator() 
            
            slam = fastSLAM2(self.parameters["slam"]["n_particles"],gt_car_traj[0,:],localization_covariance,observed_clique_ids,performance_tracker.all_data_associations[min(performance_tracker.all_data_associations.keys())])
            #start_exp,start_t,postProcessing_dir,clique_sim,slam
            print("loading in previous results...")
            if self.comparison_type == "betterTogether": 
                clique_sim,slam = load_in_previous_results(avilable_start_exp,avilable_start_tstep,self.args.postProcessing_dirName,clique_sim,slam) 
            elif self.comparison_type == "multiMap": 
                multiMap_sim,slam = load_in_previous_results_multiMap()  
            else: 
                slam = load_in_previous_results_vanilla()
            #self.parameters["betterTogether"]["lambda_u"],all_clique_feats,results_dir,self.parameters["experiments"],sim_length 
        else:
            if exp == 0 or self.performance_tracker.clique_inst is None: 
                print("this is exp: {}... initting clique simulator!".format(exp)) 
                if self.comparison_type == "betterTogether": 
                    clique_sim = betterTogether_simulator(self.parameters,all_clique_features=all_clique_feats,current_exp=exp,sim_length=sim_length,
                                                data_association=performance_tracker.all_data_associations,exp_observations=exp_observations,exp_obs_ids=observed_clique_ids,
                                                feature_detection=self.args.enable_feature_detection,verbosity=self.args.verbose,start_time=(self.args.start_experiment,self.args.start_tstep))
                elif self.comparison_type == "multiMap":
                    multiMap_sim = multiMap_simulator() 
                slam = fastSLAM2(self.parameters["slam"]["n_particles"],gt_car_traj[0,:],localization_covariance,observed_clique_ids,performance_tracker.all_data_associations[min(performance_tracker.all_data_associations.keys())]) 

            else:  
                if self.comparison_type == "betterTogether":  
                    print("need to preserve the clique instance so the posteriors don't drop between experiments!") 
                    clique_sim = self.performance_tracker.clique_inst 
                    clique_sim.reinit_experiment(exp,exp_observations,all_clique_feats,observed_clique_ids) 
                elif self.comparison_type == "multiMap": 
                    multiMap_sim = self.performance_tracker.multimap_inst 
                    multiMap.reinit_experiment() 
                slam = self.performance_tracker.slam_inst 

        if self.parameters["isCarla"]: 
            n = exp + 1 
        else: 
            n = exp

        if self.args.plotIntermediateResults:
            if exp == 0:
                plotter = betterFaster_plot(self.parameters["experiments"],exp,self.parameters,gt_car_traj,observed_clique_ids,performance_tracker.gt_gstates[n],
                                frame_dir=self.args.frame_dir,show_plots=self.args.showPlots,verbose=self.args.verbose,plot_int_results=self.args.plotIntermediateResults)
            else: 
                plotter = betterFaster_plot(self.parameters["experiments"],exp,self.parameters,gt_car_traj,observed_clique_ids,performance_tracker.gt_gstates[n],
                                prev_lm_est_err=performance_tracker.ind_landmark_estimate_error_cache,frame_dir=self.args.frame_dir,show_plots=self.args.showPlots,
                                verbose=self.args.verbose,plot_int_results=self.args.plotIntermediateResults)
            
        self.performance_tracker.init_new_experiment(exp,clique_sim,slam,gt_car_traj,self.parameters)

        processing_time = None 
        #TIMING STUFF TRYNA GO FAST 
        slam_times = {} 
        slam_times["prediction"] = []
        slam_times["correction"] = []
        clique_times = []
        perf_times = []
        plotting_times = []

        int_background_process = None 
        background_process = None 
        update_process = None 

        print("done with initialization time to go through the timesteps!!")
        #3. Iterate through the timesteps 
        for t in range(sim_length): 
            if np.mod(t,5) == 0: 
                print("this is experiment {} of {}, t: {} of {}".format(exp,self.parameters["experiments"],t,sim_length)) 

            if t < self.args.start_tstep and exp == self.args.start_experiment:
                print("we are skipping this timestep:{} as per the commandline arguments...".format(t))
                continue 

            t0 = time.time() 

            slam_time0 = time.time()
            #3a. Estimate Pose 
            estimated_pose = slam.prediction([gt_car_traj[t,0],gt_car_traj[t,1],gt_car_traj[t,2]]) #x,y,yaw
        
            observations_t = exp_observations[t] 

            slam_time1 = time.time() 

            
            #3c. update clique sim to find persistent landmarks 
            if self.comparison_type == "betterTogether": 
                clique_time0 = time.time()
                persistent_observations = clique_sim.update(t,observations_t)
                clique_time1 = time.time() 
                clique_times.append(clique_time1 - clique_time0)
            elif self.comparison_type == "multiMap": 
                persistent_observations = multiMap.update(t,observations_t)
            else: 
                persistent_observations = get_vanilla_observation_rate 

            slam_time2 = time.time() 
            #3d. SLAM correction
            slam.correction(t,persistent_observations)
            slam_time3 = time.time() 

            slam_times["prediction"].append(slam_time1-slam_time0) 
            slam_times["correction"].append(slam_time3 - slam_time2)
            #slam_times.append((slam_time1-slam_time0) + (slam_time3 - slam_time2))

            #perf_time0 = time.time() 
            #3e. update performance tracker 
            performance_tracker.update(t,clique_sim,slam,processing_time)
            #perf_time1 = time.time() 
            #perf_times.append(perf_time1 - perf_time0)

            if not update_process is None: 
                if update_process.is_alive():
                    update_process.join() 

            update_process = multiprocessing.Process(target=performance_update_background, args=(performance_tracker,t,clique_sim,slam,processing_time))
            processing_time = time.time() - t0 

            #3f. plot 
            if self.args.plotIntermediateResults:
                plot_time0 = time.time() 
                plotter.plot_state(slam,t,estimated_pose,observations_t,clique_sim.posteriors,clique_sim.growth_state_estimates)
                plot_time1 = time.time() 
                plotting_times.append(plot_time1 - plot_time0)

            if t > 10 and np.mod(t,10) == 0:
                if self.args.verbose: 
                    print("Mean SLAM prediction time: {}, Median SLAM predicition time: {}".format(np.mean(slam_times["prediction"]),np.median(slam_times["prediction"]))) 
                    print("Mean SLAM correction time: {}, Median SLAM correction time: {}".format(np.mean(slam_times["correction"]),np.median(slam_times["correction"]))) 
                    print("Mean CLIQUE time: {}, Median CLIQUE time: {}".format(np.mean(clique_times),np.median(clique_times)))
                    #print("Mean Performance time: {}, Median Performance time: {}".format(np.mean(perf_times),np.median(perf_times))) 
                    if self.args.plotIntermediateResults: 
                        print("Mean Plotting time: {}, Median Plotting time: {}".format(np.mean(plotting_times),np.median(plotting_times)))
            
            if int_background_process is not None:
                if int_background_process.is_alive(): 
                    int_background_process.join() 

            #Run async
            if t > self.args.int_pickle_frequency - 1:
                if np.mod(t,self.args.int_pickle_frequency) == 0 or t == sim_length - 1:
                    #Pickle intermediate results so we can load them in later if need be 
                    print("pickling intermediate results!")
                    int_background_process = multiprocessing.Process(target=pickle_intermediate_results_background, args=(performance_tracker,clique_sim, slam, t))
                    #performance_tracker.pickle_intermediate_results(clique_sim,slam,t)
                    int_background_process.start()
                    #input("This is the intermediate pickling step. Press Enter to Continue...")

        #4. Write results 
        if not self.args.skip_writing_results:
            if not (self.args.load_prev_exp_results and exp < self.args.start_experiment):  
                if not background_process is None: 
                    if background_process.is_alive():
                        background_process.join() 
                        
                #background_process = write_results(exp,self.parameters,performance_tracker,self.args.postProcessing_dirName)
                background_process = multiprocessing.Process(target=save_results_background,args=(exp,self.parameters,performance_tracker,self.args.postProcessing_dirName))
                background_process.start() 
            else:  
                print("skipping writing results....")

        return exp 
    