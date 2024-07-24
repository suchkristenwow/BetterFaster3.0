#import matplotlib 
#matplotlib.use('Agg') 
import numpy as np 
import pickle 
import argparse 
import os 
import time 
import toml 
import multiprocessing
from betterFaster import *
import matplotlib.pyplot as plt 

class RunBetterFaster: 
    def __init__(self,args,performance_tracker): 
        self.args = args 
        with open(self.args.config,"r") as f:
            self.parameters = toml.load(f)
        self.performance_tracker = performance_tracker
        self.already_loaded_prev_results = False 

    def run(self,exp): 
        if exp < self.args.start_experiment:
            print("we are skipping this experiment: {} as per the command line arguments...".format(exp))
            return exp 

        #1. Load in Data
        results_dir = self.parameters["results_dir"]

        sim_length = self.parameters["sim_length"]

        if self.args.load_prev_exp_results: 
            avilable_start_exp, avilable_start_tstep = get_start_time_exp(self.args.postProcessing_dirName,self.args.start_experiment,self.args.start_tstep,sim_length)
            if not self.already_loaded_prev_results: 
                if exp != avilable_start_exp or avilable_start_tstep != self.args.start_tstep:
                    print("Sorry we can't run that experiment. Going to start at exp:{},timestep:{} instead".format(avilable_start_exp,avilable_start_tstep))
                    input("Press enter to continue ...")
                exp = avilable_start_exp; self.args.start_experiment = avilable_start_exp; self.args.start_tstep = avilable_start_tstep
                self.already_loaded_prev_results = True 
                #postProcessing_dirName
                if os.path.exists(os.path.join(self.args.postProcessing_dirName,"int_results/exp"+str(exp)+"_int_performance_results"+str(avilable_start_tstep)+".pickle")): 
                    performance_tracker.load_prev_results(os.path.join(self.args.postProcessing_dirName,"int_results/exp"+str(exp)+"_int_performance_results"+str(avilable_start_tstep)+".pickle"))
                print("loaded in exp results!")

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
            obsd_clique_path = os.path.join(results_dir,"observation_pickles/exp"+str(exp)+"observed_cliques.pickle")
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

        #Convert the third column from degrees to radians 
        gt_car_traj[:,2] = np.deg2rad(gt_car_traj[:,2]) 

        localization_covariance_params = self.parameters["slam"]["localization_covariance_params"]
        var_x = localization_covariance_params[0]; var_y = localization_covariance_params[1]; var_theta = localization_covariance_params[2]
        localization_covariance = np.diag([var_x,var_y,var_theta])
        
        #if self.parameters["isCarla"]: 
        sim_utils = simUtils(exp,performance_tracker.all_data_associations,self.parameters)

        #2. Instantiate Simulator Classes 
        print("instantiating classes")
        if self.args.start_experiment > 0 and self.args.load_prev_exp_results:
            clique_sim = clique_simulator(self.parameters,all_clique_features=all_clique_feats,current_exp=exp,sim_length=sim_length,
                                            data_association=performance_tracker.all_data_associations,exp_observations=exp_observations,exp_obs_ids=observed_clique_ids,
                                            feature_detection=self.args.enable_feature_detection,verbosity=self.args.verbose,start_time=(self.args.start_experiment,self.args.start_tstep))
            #init_pose,localization_covariance,n_particles,lm_ids,Q_params
            slam = fastSLAM2(gt_car_traj[0,:],localization_covariance,self.parameters["slam"]["n_particles"],observed_clique_ids,self.parameters["slam"]["betas"])
            #start_exp,start_t,postProcessing_dir,clique_sim,slam
            print("loading in previous results...")
            clique_sim,slam = load_in_previous_results(avilable_start_exp,avilable_start_tstep,self.args.postProcessing_dirName,clique_sim,slam)
        else:
            if exp == 0 or self.performance_tracker.clique_inst is None: 
                print("this is exp: {}... initting clique simulator!".format(exp)) 
                clique_sim = clique_simulator(self.parameters,all_clique_features=all_clique_feats,current_exp=exp,sim_length=sim_length,
                                            data_association=performance_tracker.all_data_associations,exp_observations=exp_observations,exp_obs_ids=observed_clique_ids,
                                            feature_detection=self.args.enable_feature_detection,verbosity=self.args.verbose,start_time=(self.args.start_experiment,self.args.start_tstep))
                #init_pose,localization_covariance,n_particles,lm_ids,Q_params 
                slam = fastSLAM2(gt_car_traj[0,:],localization_covariance,self.parameters["slam"]["n_particles"],observed_clique_ids,self.parameters["slam"]["betas"])
            else:   
                print("need to preserve the clique instance so the posteriors don't drop between experiments!") 
                clique_sim = self.performance_tracker.clique_inst 
                clique_sim.reinit_experiment(exp,exp_observations,all_clique_feats,observed_clique_ids)
                slam = self.performance_tracker.slam_inst 

        if self.parameters["isCarla"]: 
            n = exp + 1 
        else: 
            n = exp

        #if self.args.plotIntermediateResults:
        plt.close() 
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
        n_any_observations = []
        n_persistent_observations = []
        #3. Iterate through the timesteps 
        for t in range(sim_length): 
            if np.mod(t,10) == 0: 
                print("this is experiment {} of {}, t: {} of {}".format(exp,self.parameters["experiments"],t,sim_length)) 

            if t < self.args.start_tstep and exp == self.args.start_experiment:
                print("we are skipping this timestep:{} as per the commandline arguments...".format(t))
                continue 

            t0 = time.time() 

            slam_time0 = time.time()
            #3a. Estimate Pose 
            #print("gt pose: ",[gt_car_traj[t,0],gt_car_traj[t,1],gt_car_traj[t,2]])
            estimated_pose = slam.prediction([gt_car_traj[t,0],gt_car_traj[t,1],gt_car_traj[t,2]]) #x,y,yaw
            observations_t = exp_observations[t] 

            #observations_t = reject_outliers(sim_utils,[gt_car_traj[t,0],gt_car_traj[t,1],gt_car_traj[t,2]],observations_t)

            n_any_observations.append(len(observations_t)) 
            #print("there were {} observations this timestep".format(len(observations_t))) 
            
            slam_time1 = time.time() 

            clique_time0 = time.time()
            #3c. update clique sim to find persistent landmarks 
            persistent_observations,reinit_ids = clique_sim.update(t,observations_t)

            #slam.reinit_EKFs(reinit_ids) 
            #persistent_observations = observations_t # removing the clique step for debugging 
            
            n_persistent_observations.append(len(persistent_observations)) 

            clique_time1 = time.time() 
            clique_times.append(clique_time1 - clique_time0)

            slam_time2 = time.time() 
            #3d. SLAM correction
            slam.correction(persistent_observations)

            #DEBUG 
            '''
            idx = np.argmax([x.weight for x in slam.particles]) 
            best_landmarks = slam.particles[idx].landmarks 
            for landmark in best_landmarks: 
                landmark_center = landmark.mu 
                if not np.all(landmark_center == 0): 
                    print("Best guess center estimate landmark {}: {}".format(landmark.lm_id,landmark_center))
            ''' 

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
            plotter.plot_state(slam,t,estimated_pose,observations_t,clique_sim.posteriors,clique_sim.growth_state_estimates)

            if self.args.plotIntermediateResults:
                plot_time0 = time.time() 
                plot_time1 = time.time() 
                plotting_times.append(plot_time1 - plot_time0)
                if t == sim_length - 1: 
                    #close the plots 
                    plt.close() 

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
                    #int_background_process = multiprocessing.Process(target=pickle_intermediate_results_background, args=(performance_tracker,clique_sim, slam, t))
                    performance_tracker.pickle_intermediate_results(clique_sim,slam,t)
                    #int_background_process.start()
                    #input("This is the intermediate pickling step. Press Enter to Continue...")

        #4. Write results 
        if not self.args.skip_writing_results:
            if not (self.args.load_prev_exp_results and exp < self.args.start_experiment):  
                '''
                if not background_process is None: 
                    if background_process.is_alive():
                        background_process.join() 
                '''
                write_results(exp,self.parameters,performance_tracker,self.args.postProcessing_dirName)
                #background_process = multiprocessing.Process(target=save_results_background,args=(exp,self.parameters,performance_tracker,self.args.postProcessing_dirName))
                #background_process.start() 
            else:  
                print("skipping writing results....")

        return exp 
    
if __name__ == "__main__": 
    #/media/kristen/easystore/BetterFaster/kitti_carla_simulator/exp_results

    parser = argparse.ArgumentParser(description="Experiment Runner")
    parser.add_argument(
        "--config",
        default="configs/fake.toml",
        type=str,
        help="Path to the TOML configuration file",
    ) 

    parser.add_argument(
        "--load_prev_exp_results",
        default=False,
        type=bool, 
        help="Load in previous experimental results (set True if you want to launch from the middle)",
    ) 

    parser.add_argument(
        "--start_experiment",
        default=0,
        type=int,
        help="Specify what experiment to start on (if you want to start from the middle)",
    ) 

    parser.add_argument(
        "--start_tstep",
        default=0,
        type=int,
        help="Specify what timestep to start on (if you want to start from the middle)",
    ) 

    parser.add_argument(
        "--skip_writing_results",
        default=False,
        type=bool,
        help="Write results? (set True if you dont care about saving the results)",
    ) 

    parser.add_argument(
        "--frame_dir",
        default=None,
        type=str,
        help="This is where you want to save all the frames of all the plots to make animations later",
    ) 

    parser.add_argument(
        "--plotIntermediateResults",
        default=False,
        type=bool,
        help="This is will plot everything as you go. Will take a lot longer",
    ) 

    parser.add_argument(
        "--postProcessing_dirName",
        type=str,
        help="This is where all the results will be saved",
        required=True 
    ) 

    parser.add_argument(
        "--showPlots",
        type=bool,
        default=False,
        help="If you want to watch the plots in real time. This will slow everything down A LOT, just syk"
    ) 

    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Debugging Mode will spam so much info"
    ) 

    parser.add_argument(
        "--enable_feature_detection",
        default = True, 
        help = "This is for debugging bc I'm sus of the feature detection. False sets all features to be detected."
    )

    parser.add_argument(
        "--int_pickle_frequency",
        type=int,
        default = 25, 
        help = "How frequently do you want to save results."
    )

    args = parser.parse_args() 
    if not os.path.exists(args.postProcessing_dirName): 
        print("creating: {}".format(args.postProcessing_dirName))
        os.mkdir(args.postProcessing_dirName)

    with open(args.config,"r") as f:
        parameters = toml.load(f)
    #Instantiate Performance Class...  this tracks all the relevant statistics between experiments
    
    performance_tracker = PerformanceTracker(parameters,args.postProcessing_dirName) 

    '''
    comparison_runner_insts = []
    for data_association_rate in parameters['comparison_params']['data_association_rates']:
        out_dir = os.path.join(args.postProcessing_dirName,comparison_type + "_" + str(data_association_rate))
        if parameters['comparison_params']['compare_betterTogether']: 
            betterTogether_performance_tracker = PerformanceTracker(parameters,out_dir,comparison_type="betterTogether")
            comparison_runner_insts.append(RunComparison(args,betterTogether_performance_tracker,data_association_rate,comparison_type="betterTogether")) 
        if parameters['comparison_params']['compare_vanilla']: 
            vanilla_performance_tracker = PerformanceTracker(parameters,out_dir,comparison_type="vanilla")
            comparison_runner_insts.append(RunComparison(args,vanilla_performance_tracker,data_association_rate,comparison_type="vanilla"))
        if parameters['comparison_params']['compare_multiMap']: 
            multiMap_performance_tracker = PerformanceTracker(parameters,out_dir,comparison_type="multiMap") 
            comparison_runner_insts.append(RunComparison(args,multiMap_performance_tracker,data_association_rate,comparison_type="multiMap")) 
    '''

    runner = RunBetterFaster(args,performance_tracker)

    exp_ran = None 
    for exp in range(parameters["experiments"]):
        if args.load_prev_exp_results:
            if exp_ran is not None and exp <= exp_ran:
                print("skipping exp: {} as per command line arguments".format(exp))
                continue 
        exp_ran = runner.run(exp)  
        '''
        #TO DO: RUN THIS IN PARALLEL 
        for comparison_method in comparison_runner_insts: 
            comparison_method.run(exp) 
        '''