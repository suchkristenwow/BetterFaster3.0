import numpy as np 
import pickle 
import argparse 
import os 
import time 
import toml 
from betterFaster import *

class RunBetterFaster: 
    def __init__(self,args,performance_tracker): 
        self.args = args 
        with open(self.args.config,"r") as f:
            self.parameters = toml.load(f)
        self.performance_tracker = performance_tracker

    def run(self,exp): 
        if exp < self.args.start_experiment:
            print("we are skipping this experiment: {} as per the command line arguments...".format(exp))
            return 

        #1. Load in Data
        results_dir = self.parameters["results_dir"]

        sim_length = get_sim_length(results_dir,exp)

        clique_feats_path = os.path.join(results_dir,"exp"+str(exp + 1)+"all_clique_feats.pickle")
        with open(clique_feats_path,"rb") as handle:
            all_clique_feats = pickle.load(handle)

        observed_clique_ids = get_observed_clique_ids(exp,all_clique_feats,self.performance_tracker)

        obsd_clique_path = os.path.join(results_dir,"observation_pickles/exp"+str(exp + 1)+"observed_cliques.pickle")
        with open(obsd_clique_path,"rb") as handle:
            exp_observations = pickle.load(handle)

        assert len(exp_observations) > 0, "No observations for this experiment?"

        if not self.parameters["isCarla"]:
            gt_car_traj = np.genfromtxt(os.path.join(self.parameters["results_dir"],"fake_gt_traj.csv"),delimiter=",")
        '''
        else:
            TO DO: experimental version 
        '''

        localization_covariance_params = self.parameters["slam"]["localization_covariance_params"]
        var_x = localization_covariance_params[0]; var_y = localization_covariance_params[1]; var_theta = localization_covariance_params[2]
        localization_covariance = np.diag([var_x,var_y,var_theta])

        #2. Instantiate Simulator Classes 
        if self.parameters["isCarla"]: 
            sim_utils = simUtils(exp,performance_tracker.all_data_associations[exp + 1],self.parameters)

        if exp == 0 or self.performance_tracker.clique_inst is None: 
            print("this is exp: {}... initting clique simulator!".format(exp))
            clique_sim = clique_simulator(self.parameters,all_clique_features=all_clique_feats,current_exp=exp,sim_length=sim_length,
                                          data_association=performance_tracker.all_data_associations,exp_observations=exp_observations)
            slam = fastSLAM2(self.parameters["slam"]["n_particles"],gt_car_traj[0,:],localization_covariance,observed_clique_ids,performance_tracker.all_data_associations[exp + 1])
        else:   
            #need to preserve the clique instance so the posteriors don't drop between experiments 
            clique_sim = self.performance_tracker.clique_inst 
            clique_sim.reinit_experiment(exp,exp_observations)
            slam = self.performance_tracker.slam_inst 
            if self.parameters["comparison_bools"]["compare_betterTogether"]: 
                data_association_rates = self.parameters["betterFaster"]["data_association_rates"]
                betterTogether_sims = {}; untuned_slams = {} 
                for data_association_rate in data_association_rates: 
                    betterTogether_sims[data_association_rate] = self.performance_tracker.comparison_sim_instance["untuned"+str(data_association_rate)]
                    betterTogether_sims[data_association_rate].reinit_experiment(exp,exp_observations)
                    untuned_slams[data_association_rate] = self.performance_tracker.comparison_slam_instance["untuned"+str(data_association_rate)]

        if exp == 0:
            plotter = betterFaster_plot(self.parameters["experiments"],exp,self.parameters,gt_car_traj,observed_clique_ids,performance_tracker.gt_gstates[exp],
                            frame_dir=self.args.frame_dir,show_plots=self.args.showPlots,verbose=self.args.verbose)
        else: 
            plotter = betterFaster_plot(self.parameters["experiments"],exp,self.parameters,gt_car_traj,observed_clique_ids,performance_tracker.gt_gstates[exp],
                            prev_lm_est_err=performance_tracker.ind_landmark_estimate_error_cache,frame_dir=self.args.frame_dir,show_plots=self.args.showPlots,verbose=self.args.verbose)
            
        self.performance_tracker.init_new_experiment(exp,clique_sim,slam,gt_car_traj,self.parameters)

        if self.parameters["comparison_bools"]["compare_betterTogether"]: 
            self.performance_tracker.init_new_comparisons(self.parameters)

        processing_time = None 
        #TIMING STUFF TRYNA GO FAST 
        slam_times = []
        clique_times = []
        perf_times = []
        plotting_times = []
        #TIMING STUFF TRYNA GO FAST 
        #3. Iterate through the timesteps 
        for t in range(sim_length): 
            print("this is experiment {} of {}, t: {} of {}".format(exp,self.parameters["experiments"],t,sim_length)) 
            if t < self.args.start_tstep and exp == self.args.start_experiment:
                print("we are skipping this timestep:{} as per the commandline arguments...".format(t))
                continue 

            t0 = time.time() 

            slam_time0 = time.time()
            #3a. Estimate Pose 
            if self.parameters["isCarla"]: 
                estimated_pose = slam.prediction([gt_car_traj[t,0],gt_car_traj[t,1],gt_car_traj[t,5]]) #x,y,yaw
                carla_observations_t = exp_observations[t] 
                #3b. (optional) parse observations 
                observations_t = sim_utils.reform_observations(t,np.array([gt_car_traj[t,0],gt_car_traj[t,1],gt_car_traj[t,5]]),carla_observations_t) 
            else:
                estimated_pose = slam.prediction([gt_car_traj[t,0],gt_car_traj[t,1],gt_car_traj[t,2]]) #x,y,yaw
                #print("estimated_pose: ",estimated_pose)
                #traj_err = np.linalg.norm(estimated_pose[:2] - gt_car_traj[t,:2])
                observations_t = exp_observations[t] 
            slam_time1 = time.time() 

            clique_time0 = time.time()
            #3c. update clique sim to find persistent landmarks 
            persistent_observations = clique_sim.update(t,observations_t)
            clique_time1 = time.time() 
            clique_times.append(clique_time1 - clique_time0)

            slam_time2 = time.time() 
            #3d. SLAM correction
            slam.correction(t,persistent_observations)
            slam_time3 = time.time() 

            slam_times.append((slam_time1-slam_time0) + (slam_time3 - slam_time2))

            perf_time0 = time.time() 
            #3e. update performance tracker 
            performance_tracker.update(t,clique_sim,slam,processing_time)
            perf_time1 = time.time() 
            perf_times.append(perf_time1 - perf_time0)

            processing_time = time.time() - t0 
            #3f. plot 
            plot_time0 = time.time() 
            plotter.plot_state(slam,t,estimated_pose,observations_t,clique_sim.posteriors,clique_sim.growth_state_estimates)
            plot_time1 = time.time() 
            plotting_times.append(plot_time1 - plot_time0)

            if t > 10 and np.mod(t,10) == 0:
                print("Mean SLAM time: {}, Median SLAM time: {}".format(np.mean(slam_times),np.median(slam_times))) 
                print("Mean CLIQUE time: {}, Median CLIQUE time: {}".format(np.mean(clique_times),np.median(clique_times)))
                print("Mean Performance time: {}, Median Performance time: {}".format(np.mean(perf_times),np.median(perf_times)))
                print("Mean Plotting time: {}, Median Plotting time: {}".format(np.mean(plotting_times),np.median(plotting_times)))

        #4. Write results 
        if not self.args.skip_writing_results: 
            print("self.args.postProcessing_dirName: ",self.args.postProcessing_dirName)
            print("self.args: ",self.args)
            #experiment_no,results_dir,performance_tracker
            write_results(exp,self.parameters,performance_tracker,self.args.postProcessing_dirName)

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
        "--postProcessing_dirName",
        type=str,
        help="This is where all the results will be saved",
        required=True 
    ) 

    parser.add_argument(
        "--showPlots",
        type=bool,
        default=False,
        help="If you want to watch the plots in real time. This will slow everything down, just syk"
    ) 

    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Debugging Mode will spam"
    ) 

    args = parser.parse_args() 
    if not os.path.exists(args.postProcessing_dirName): 
        print("creating: {}".format(args.postProcessing_dirName))
        os.mkdir(args.postProcessing_dirName)

    with open(args.config,"r") as f:
        parameters = toml.load(f)
    #Instantiate Performance Class...  this tracks all the relevant statistics between experiments
    performance_tracker = PerformanceTracker(parameters)
    runner = RunBetterFaster(args,performance_tracker)
    for exp in range(parameters["experiments"]):
        runner.run(exp)