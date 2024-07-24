import os 
import pickle 
import matplotlib.pyplot as plt 
import numpy as np 
'''
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
'''

out_dir = "/home/kristen/fake_postProcessing_results"
posterior_dir = os.path.join(out_dir,"posteriors")  
# plot posteriors
sim_length = 500 
for exp in range(10):
    t0 = exp*sim_length; tf = (exp+1)*sim_length 
    with open(os.path.join(posterior_dir,"exp0.pickle"),"rb") as handle:
        posterior_pickle = pickle.load(handle)
        #print("posterior_pickle: ",posterior_pickle) 
        fig,axs = plt.subplots(len(posterior_pickle.keys()),1,figsize=(8,12))  
        fig.suptitle(f"Posteriors: Experiment {exp + 1}") 
        for i,id_ in enumerate(posterior_pickle.keys()):
            axs[i].plot(np.arange(5000),posterior_pickle[id_])  
            axs[i].set_title(f"Clique {id_}") 
        plt.tight_layout()  
        fig.savefig("posterior_exp"+str(exp)+".png") 
        plt.close() 

#frame_dir = "/media/kristen/easystore/BetterFaster/kitti_carla_simulator/exp_results/plotFrames" 
# make animation 

# plot gstate estimates 
#load intermediate results 
#int_perf_pickle["gstate_error_cache"] = self.gstate_estimate_error_cache 

# plot trajectory error + plot lm estimate error 
experiment_no = 10
fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # 2 Rows, 1 Column 
fig.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.05)

with open(os.path.join(out_dir,"int_results/exp"+str(experiment_no - 1)+"_int_performance_results499.pickle"),"rb") as handle:
    perf_results = pickle.load(handle) 
#TRAJECTORY
best_traj_estimate = perf_results["trajectory_estimate_error"]
#axs[0].plot(performance_tracker.trajectory_estimate_error[:last_t,0],performance_tracker.trajectory_estimate_error[:last_t,1],'r')
axs[0].plot(np.arange((experiment_no)*sim_length),best_traj_estimate[:,1],'r')
axs[0].set_title('Trajectory Error',ha='left',va='center',x=-0.2,y=0.5,rotation=90)
axs[0].set_xlabel('Time')
axs[0].set_ylabel('MSE (m)')
axs[0].set_xlim(0,(experiment_no)*sim_length) 
axs[0].set_ylim(0,max(best_traj_estimate[:,1])*1.05)
axs[0].grid(True)

#LANDMARK 
'''
for exp in range(experiment_no + 1): 
    with open(os.path.join(out_dir,"int_results/exp"+str(exp)+"_int_performance_results499.pickle"),"rb") as handle:
        perf_results = pickle.load(handle)
    first_t = exp * sim_length 
    last_t = (exp + 1)*sim_length 
    axs[1].plot(np.arange(first_t,last_t),perf_results["landmark_estimate_error_cache"]["mean"][first_t:last_t],'b')  
'''
axs[1].plot(np.arange(sim_length*(experiment_no)),perf_results["landmark_estimate_error_cache"]["mean"],'b')
last_t = (experiment_no)*sim_length 
axs[1].set_title('Mean Landmark \nEstimate Error',ha='left',va='center',x=-0.2,y=0.5,rotation=90)
axs[1].set_xlabel('Time')
axs[1].set_ylabel('MSE (m)')
axs[1].set_xlim(0,last_t)
#print("perf_results[landmark_estimate_error_cache][:last_t,1]*1.05: ",perf_results["landmark_estimate_error_cache"][:last_t,1]*1.05)
axs[1].set_ylim(0,max(perf_results["landmark_estimate_error_cache"]["mean"])*1.05) 
axs[1].grid(True)

#OBSERVATIONS
'''
results_dir = "/media/kristen/easystore1/BetterFaster/kitti_carla_simulator/exp_results" 
n_observations = []
for exp in range(11): 
    with open(os.path.join(results_dir,"reformed_carla_observations/exp"+str(exp)+"reformed_carla_observations.pickle"),"rb") as handle:
        exp_observations = pickle.load(handle)
    for t in range(1000): 
        observations_t = exp_observations[t] 
        n_observations.append(len(observations_t)) 
axs[2].plot(np.arange(len(n_observations)),n_observations) 
axs[2].set_xlabel('Time')
axs[2].set_title('# of \nObservations',ha='left',va='center',x=-0.2,y=0.5,rotation=90) 
axs[2].set_xlim(0,last_t)
'''
results_dir = "/home/kristen/BetterFaster3.1/betterFaster/sim_utils/fake_data/observation_pickles"
n_observations = [] 
for exp in range(10):
    with open(os.path.join(results_dir,"exp"+str(exp)+"observed_cliques.pickle"),"rb") as handle:
        exp_observations = pickle.load(handle) 
    for t in range(sim_length): 
        observations_t = exp_observations[t] 
        n_observations.append(len(observations_t)) 
axs[2].plot(np.arange(len(n_observations)),n_observations) 
axs[2].set_xlabel('Time') 
axs[2].set_title('# of \nObservations',ha='left',va='center',x=-0.2,y=0.5,rotation=90) 
axs[2].set_xlim(0,last_t)

#plt.tight_layout() 
filename = "experiment"+str(experiment_no)+"_slam_err_plt.jpg" 
if not os.path.exists(os.path.join(out_dir,"slam_err_plots")): 
    os.mkdir(os.path.join(out_dir,"slam_err_plots"))
plt.savefig(os.path.join(out_dir,"slam_err_plots/" + filename))
plt.show(block=True) 
#plt.close() 