import numpy as np
import sys
import os 
import pickle 
sys.path.append('betterTogether')
sys.path.append('sim_utils')
sys.path.append('plotting')
sys.path.append('SLAMCore')
from CliqueSim import clique_simulator
from fastSLAM2 import fastSLAM2
from plot_utils import betterFaster_plot
from performance import PerformanceTracker
from utils import simUtils, get_sim_length, get_reinitted_id 
from extract_gt_car_traj import get_gt_car_data
from extract_all_possible_observations import get_second_most_recent
import argparse 

import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

def run_betterFaster(experiment_no,performance_tracker,**kwargs):
	#performance_tracker = kwargs.get("performance_tracker")
	P_Miss_detection = kwargs.get('P_Miss_detection')
	P_False_detection = kwargs.get('P_False_detection')
	detection_threshold = kwargs.get('detection_threshold')
	rejection_threshold = kwargs.get('rejection_threshold')
	compare_betterTogether = kwargs.get("compare_betterTogether")
	compare_vanilla = kwargs.get("compare_vanilla")
	compare_multiMap = kwargs.get("compare_multiMap")
	sensor_noise_variance = kwargs.get('sensor_noise_variance')
	confidence_range = kwargs.get('confidence_range')
	#miss_detection_probability_function = kwargs.get('miss_detection_probability_function') 
	#localization_covariance = kwargs.get('localization_covariance')
	miss_detection_probability_function = lambda d: -2*10**(-6)*d**2 + .0025*d 
	localization_covariance = np.genfromtxt("rand_localization_covariance.csv")
	skip_writing_files = kwargs.get('skip_writing_files')
	min_feats = kwargs.get('min_feats')
	max_feats = kwargs.get('max_feats')
	#all_results_dir = kwargs.get('results_dir')
	n_particles = kwargs.get('n_particles')
	lambda_u = kwargs.get('lambda_u')
	complete_experimantes = kwargs.get("complete_experiments")
	last_t = kwargs.get("last_t")
	#camera params 
	width = kwargs.get("img_width")
	height = kwargs.get("img_height")
	fov = kwargs.get("fov")
	data_association_rates = [0.3,0.6,0.9]

	all_results_dir = "/media/arpg/easystore1/BetterFaster/kitti_carla_simulator/"
	if not os.path.exists(os.path.join(all_results_dir,"exp1_results")): 
		#print(os.path.join(all_results_dir,"exp"+str(experiment_no)+"_results"))
		all_results_dir = "/media/arpg/easystore/BetterFaster/kitti_carla_simulator/"
		if not os.path.exists(os.path.join(all_results_dir,"exp1_results")):
			print(os.path.join(all_results_dir,"exp1_results"))
			raise OSError
		
	results_dir = os.path.join(all_results_dir,"exp1_results")

	sim_length = get_sim_length(results_dir,experiment_no)

	#all_clique_feats[lm_id][feature id]["feat des"]/["feat_loc"]
	if not complete_experimantes:
		print("loading in ",os.path.join("sim_utils/intermediate_pickles",str(last_t) + "experiment"+ str(experiment_no)+ "all_clique_feats.pickle"))
		with open(os.path.join("sim_utils/intermediate_pickles",str(last_t) + "experiment"+ str(experiment_no)+ "all_clique_feats.pickle"),"rb") as handle:
			#pickle.dump(all_clique_feats,handle,protocol=pickle.HIGHEST_PROTOCOL)
			all_clique_feats = pickle.load(handle)
	else:
		with open(os.path.join(results_dir,"observation_pickles/experiment"+ str(experiment_no)+ "all_clique_feats.pickle"),"rb") as handle:
			all_clique_feats = pickle.load(handle)

	all_observed_clique_ids = [x for x in all_clique_feats.keys()] #([0, 4140, 4203, 4148, 4211])
	observed_clique_ids = []
	for id_ in all_observed_clique_ids: 
		#all_data_associations,n,id_
		orig_id = get_reinitted_id(performance_tracker.all_data_associations,experiment_no,id_)
		if orig_id not in observed_clique_ids:
			observed_clique_ids.append(orig_id)

	#obsd_cliques[exp][t] = observations
	#	observations[lm_id][feature_id]["feat_des"]
	#	observations[lm_id][feature_id]["feat_loc"]

	if not complete_experimantes:
		with open(os.path.join("sim_utils/intermediate_pickles",str(last_t) + "experiment"+ str(experiment_no)+ "observed_cliques.pickle"),"rb") as handle:
			#pickle.dump(obsd_cliques,handle,protocol=pickle.HIGHEST_PROTOCOL)
			obsd_cliques = pickle.load(handle)
	else:
		with open(os.path.join(results_dir,"observation_pickles/experiment"+ str(experiment_no)+ "observed_cliques.pickle"),"rb") as handle:
			obsd_cliques= pickle.load(handle)
			#print("obsd_cliques.keys():",obsd_cliques.keys())

	exp_observations = obsd_cliques[experiment_no -1]

	sim_utils = simUtils(results_dir,experiment_no,width,height,fov,miss_detection_probability_function,sensor_noise_variance,data_association=performance_tracker.all_data_associations)

	#initialize the sim 
	if experiment_no == 1:
		clique_sim = clique_simulator(P_Miss_detection=P_Miss_detection,P_False_detection=P_False_detection,clique_features=all_clique_feats,observations=exp_observations,lambda_u=lambda_u,
			acceptance_threshold=detection_threshold,rejection_threshold=rejection_threshold,min_feats=min_feats,max_feats=max_feats,sim_length=sim_length,confidence_range=confidence_range,tune=True)
		
		if compare_betterTogether:
			betterTogether_sims = {}
			for data_association_rate in data_association_rates:
				betterTogether_sims[data_association_rate] = clique_simulator(P_Miss_detection=P_Miss_detection,P_False_detection=P_False_detection,clique_features=all_clique_feats,observations=exp_observations,lambda_u=lambda_u,
					acceptance_threshold=detection_threshold,rejection_threshold=rejection_threshold,min_feats=min_feats,max_feats=max_feats,sim_length=sim_length,confidence_range=confidence_range,tune=False)
			

	else: 
		clique_sim = performance_tracker.clique_inst 
		#reinit_experiment(self,exp,observations):
		clique_sim.reinit_experiment(experiment_no,exp_observations)
		if compare_betterTogether:
			betterTogether_sims = {}
			for data_association_rate in data_association_rates:
				betterTogether_sims[data_association_rate] = performance_tracker.comparison_sim_instance["untuned"+str(data_association_rate)]
				betterTogether_sims[data_association_rate].reinit_experiment(experiment_no,exp_observations)
			#untuned_clique_sim = performance_tracker.comparison_sim_instance["untuned"]
			#untuned_clique_sim.reinit_experiment(experiment_no,exp_observations)

	'''
	if compare_multiMap:
		#init multimap 
		raise OSError 
	
	if compare_vanilla:
		if experiment_no == 1:

		else:
	'''

	#n_particles,init_state_mean,init_state_covar,lm_ids
	if experiment_no == 1:
		slam = fastSLAM2(n_particles,localization_covariance,observed_clique_ids)
	else: 
		slam = performance_tracker.slam_inst 
		if compare_betterTogether: 
			untuned_slams = {}
			for data_association_rate in data_association_rates:
				untuned_slams[data_association_rate] = performance_tracker.comparison_slam_instance["untuned"]

	#gt_car array 
	carla_txt_file = os.path.join(results_dir,"gt_car_poses/experiment"+str(experiment_no)+"_gt_car_pose.txt")
	gt_car_traj = get_gt_car_data(carla_txt_file,sim_length)

	#initialize plot class; exp,sim_length,results_dir,gt_car_traj
	plotter = betterFaster_plot(experiment_no,sim_length,results_dir,gt_car_traj,observed_clique_ids,compare_betterTogether,compare_multiMap,compare_vanilla)

	performance_tracker.init_new_experiment(experiment_no,clique_sim,slam,compare_betterTogether,compare_multiMap,compare_vanilla)

	if compare_betterTogether:
		performance_tracker.init_new_comparisons(experiment_no,betterTogether_sims,untuned_slams)

	'''
	if compare_multiMap:
	if compare_vanilla:
	'''
	
	if complete_experimantes:
		tsteps = sim_length 
	else: 
		tsteps = last_t 

	for t in range(tsteps): 
		print("this is t: ",t)

		#print("gt_car_traj: ",[gt_car_traj[t,0],gt_car_traj[t,1],gt_car_traj[t,5]])
		estimated_pose = slam.prediction([gt_car_traj[t,0],gt_car_traj[t,1],gt_car_traj[t,5]]) #x,y,yaw
		if np.any(np.equal(estimated_pose, None)): 
			raise OSError

		#get detections 
		carla_observations_t = exp_observations[t]

		observations_t = sim_utils.reform_observations(t,np.array([gt_car_traj[t,0],gt_car_traj[t,1],gt_car_traj[t,5]]),carla_observations_t) #this puts the detections in the desired format

		persistent_observations = clique_sim.update(t,observations_t)

		'''
		if compare_betterTogether:
			untuned_observations = 
			untuned_observations = untuned_clique_sim.update(t,observations_t)

		if compare_multiMap:
			multimap_observations = 
			multiMap.update(t,multimap_observations)

		if compare_vanilla: 
			vanilla_observations = 
		'''

		slam.correction(t,persistent_observations)
		performance_tracker.update(t,clique_sim,slam)

		plotter.plot_state(t,estimated_pose,observations_t,clique_sim.posteriors)
		plt.pause(0.05)
		
		
	if not skip_writing_files:
		#save posteriors
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


def parse_arguments():
	'''
	compare_betterTogether = kwargs.get("compare_betterTogether")
	compare_vanilla = kwargs.get("compare_vanilla")
	compare_multiMap = kwargs.get("compare_multiMap")
	'''
	parser = argparse.ArgumentParser(description="Simulation parameters")
	parser.add_argument("--P_Miss_detection", type=float, default=0.1, help="Probability of miss detection")
	parser.add_argument("--P_False_detection", type=float, default=0.05, help="Probability of false detection")
	parser.add_argument("--lambda_u", type=float, default=1/350.0, help="lambda_u parameter")
	parser.add_argument("--detection_threshold", type=float, default=0.9, help="Detection threshold")
	parser.add_argument("--rejection_threshold", type=float, default=0.5, help="Rejection threshold")
	parser.add_argument("--experiments", type=int, default=12, help="Number of experiments")
	parser.add_argument("--experiments_season", type=int, default=3, help="Experiments for seasons")
	#parser.add_argument("--sim_length", type=int, default=5000, help="Simulation length (number of timesteps)")
	parser.add_argument("--min_feats", type=int, default=3, help="Minimum number of features")
	parser.add_argument("--max_feats", type=int, default=100, help="Maximum number of features")
	parser.add_argument("--sensor_noise_variance", type=float, default=0.1, help="Sensor noise variance")
	parser.add_argument("--confidence_range", type=int, default=50, help="Confidence range")
	parser.add_argument("--negative_supression", type=bool, default=True)
	parser.add_argument("--skip_writing_files", type=bool, default=True, help="Skip writing files")
	#parser.add_argument("--experiment_no", type=int, default=1)
	parser.add_argument("--n_particles", type=int, default=10)
	parser.add_argument("--complete_experiments", type=bool, default=True)
	parser.add_argument("--last_t",type=int,default=None)
	parser.add_argument("--img_width",type=int,default=1392)
	parser.add_argument("--img_height",type=int,default=1024)
	parser.add_argument("--fov",type=int,default=72)
	parser.add_argument("--compare_betterTogether",type=bool,default=False)
	parser.add_argument("--compare_vanilla",type=bool,default=False)
	parser.add_argument("--compare_multiMap",type=bool,default=False)
	return parser.parse_args()

if __name__ == "__main__":
	'''
	### SIM PARAMETERS ###    
	P_Miss_detection = 0.1
	P_False_detection = 0.05

	### SIM PARAMETERS ### 
	#lambda_u - .001, .005, .01
	lambda_u = 1/350.0 #.0029

	detection_threshold = 0.9 #0.5,0.75,0.9
	rejection_threshold = 0.5 #0.25,0.5,0.75

	experiments = 1 #max = 12 

	experiments_season = 3 #experiments for seasons

	sim_length = 500 #This is the number of timesteps you want to run (max = 500)

	min_feats = 3
	max_feats = 100 

	sensor_noise_variance = 0.1 #0.05,0.1,0.25

	confidence_range = 50 

	#self.miss_detection_probability_function = lambda d: 1 - np.exp(-1.0/self.confidence_range * d)
	miss_detection_probability_function = lambda d: -2*10**(-6)*d**2 + .0025*d 

	negative_supression = True

	skip_writing_files = False #Change to True if you dont want to save the results 

	img_width = 1392; img_height = 1024 
	fov = 72 

	localization_covariance = np.random.uniform(-2, 2, size=(3, 3))
	'''
	
	args = parse_arguments()

	performance_tracker = PerformanceTracker(args.experiments)

	'''
	run_betterFaster(experiment_no=1,results_dir=results_dir,P_Miss_detection=P_Miss_detection,P_False_detection=P_False_detection,
		detection_threshold=detection_threshold,rejection_threshold=rejection_threshold,sim_length=sim_length,
		sensor_noise_variance=sensor_noise_variance, confidence_range=confidence_range,miss_detection_probability_function=miss_detection_probability_function,
		localization_covariance=localization_covariance,skip_writing_files=skip_writing_files,min_feats=min_feats,max_feats=max_feats,n_particles=10,
		img_width=img_width,img_height=img_height,fov=fov,performance_tracker=performance_tracker,lambda_u=lambda_u,complete_experiments=False,last_t=1000)
	'''
	for exp in range(args.experiments +1): 
		run_betterFaster(exp+1,performance_tracker,**vars(args))