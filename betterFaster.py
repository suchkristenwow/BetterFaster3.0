import numpy as np 
import pickle 
import argparse 
import os 
import time 
import toml 

class RunBetterFaster: 
    def __init__(self,args,performance_tracker): 
        self.args = args 
        with open(self.args.config,"r") as f:
            self.parameters = toml.load(f)
        self.performance_tracker = performance_tracker

    def run(self,exp): 
        #1. Load in Data
        results_dir = self.parameters["results_dir"]

        sim_length = get_sim_length(results_dir,exp)

        clique_feats_path = os.path.join(results_dir,"exp_"+str(exp)+"_all_clique_feats.pickle")
        with open(clique_feats_path,"rb") as handle:
            all_clique_feats = pickle.load(handle)

        observed_cliques = get_observed_cliques(exp,self.performance_tracker)

        obsd_clique_path = os.path.join(results_dir,"exp"+str(exp)+"_observations.pickle")
        with open(obsd_clique_path,"rb") as handle:
            exp_observations = pickle.load(handle)

        assert len(exp_observations) > 0, "No observations for this experiment?"

        if not self.parameters["isCarla"]:
            gt_car_traj = np.genfromtxt(os.path.join(self.parameters["results_dir"],"fake_gt_traj.csv"),delimiter=",")

        #2. Instantiate Simulator Classes 
        if self.parameters["isCarla"]: 
            sim_utils = simUtils(exp,performance_tracker.all_data_associations[exp],self.parameters)

        if exp == 0: 
            clique_sim = clique_simulator(self.parameters,all_clique_features=all_clique_feats,current_exp=exp,sim_length=sim_length,
                                          data_association=performance_tracker.all_data_associations[exp])
            slam = fastSLAM2(self.parameters["slam"]["n_particles"])
        else:   
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

        plotter =  betterFaster_plot(exp,self.parameters,gt_car_traj,[x for x in all_clique_feats.keys()])

        self.performance_tracker.init_new_experiment(self.parameters)

        if self.parameters["comparison_bools"]["compare_betterTogether"]: 
            self.performance_tracker.init_new_comparisons(self.parameters)

        processing_time = None 
        #3. Iterate through the timesteps 
        for t in range(sim_length): 
            t0 = time.time() 
            #3a. Estimate Pose 
            if self.parameters["isCarla"]: 
                estimated_pose = slam.prediction([gt_car_traj[t,0],gt_car_traj[t,1],gt_car_traj[t,5]]) #x,y,yaw
                carla_observations_t = exp_observations[t] 
                #3b. (optional) parse observations 
                observations_t = sim_utils.reform_observations(t,np.array([gt_car_traj[t,0],gt_car_traj[t,1],gt_car_traj[t,5]]),carla_observations_t) 
            else:
                estimated_pose = slam.prediction([gt_car_traj[t,0],gt_car_traj[t,1],gt_car_traj[t,2]]) #x,y,yaw
                observations_t = exp_observations[t] 
            #3c. update clique sim to find persistent landmarks 
            persistent_observations = clique_sim.update(t,observations_t)
            #3d. SLAM correction
            slam.correction(t,persistent_observations)
            #3e. update performance tracker 
            performance_tracker.update(t,clique_sim,slam,processing_time)
            processing_time = time.time() - t0 
            #3f. plot 
            plotter.plot_state(slam,t,estimated_pose,observations_t,clique_sim.posteriors)

        #4. Write results 
        if not self.args.skip_writing_files: 
            write_results(self.parameters,performance_tracker)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Experiment Runner")
    parser.add_argument(
        "--config",
        default="configs/fake.toml",
        help="Path to the TOML configuration file",
    ) 
    parser.add_argument(
        "--load_prev_exp_results",
        default=False,
        help="Load in previous experimental results (set True if you want to launch from the middle)",
    ) 
    parser.add_argument(
        "--skip_writing_results",
        default=False,
        help="Write results? (set True if you dont care about saving the results)",
    ) 
    args = parser.parse_args 
    with open(args.config,"r") as f:
        parameters = toml.load(f)
    #Instantiate Performance Class...  this tracks all the relevant statistics between experiments
    performance_tracker = PerformanceTracker(parameters)
    runner = RunBetterFaster(args,performance_tracker)
    for exp in range(parameters["experiments"]):
        runner.run(exp)