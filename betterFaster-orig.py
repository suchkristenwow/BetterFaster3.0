import numpy as np
import sys
import os 
sys.path.append("betterFaster_utils")
sys.path.append("ORB_feat_extraction")
sys.path.append("comparison_methods")
import JointPersistenceFilter
from CliqueSim import clique_simulator
from fastSLAM2 import fastSLAM2
from get_gt_carla_landmarks import *
from observable_features_carla import *
import time
import pickle 
from write_files import *
from orb_feat_extractor import extract_carla_features
from Performance_class import Performance
from multimap_sim import multiMap_simulator


def run_betterFaster(P_Miss_detection,P_False_detection,lambda_u,detection_threshold,rejection_threshold,experiments,experiments_season,
		sim_length,sensor_noise_variance,confidence_range,miss_detection_probability_function,min_feats,max_feats,base_dir,map_name,
		multimap_params,max_dist=250,data_association_rates=[0.25,0.5,0.75,0.9,0.95],skip_writing_files=False,betterFaster=True,multimap=True,vanilla=True,
		is_visual=True,negative_supression=True,n_particles=10):

	results_dir = base_dir+map_name+"/"

	orb_dir = "/home/kristen/BetterFaster2.0/ORB_feat_extraction/"
	print("initializing clique sim!")

	max_sim_length = get_max_sim_length(results_dir)

	init_state_covar = np.identity(3)

	gt_traj = parse_carla_path(results_dir,experiments,sim_length) #need this to get the observable features
	exp_gt_traj = parse_exp_carla_path(results_dir,experiments,sim_length) 
	if not os.path.exists(orb_dir + map_name + "_orb_feats/experiment"+str(experiments-1)+"_"+str(sim_length -1).zfill(4)+".csv"):
		extract_carla_features(map_name,results_dir + "rgb_images/",experiments,sim_length)

	### Load all possibly observable features ###
	if os.path.exists("parallel_pickles/"+map_name+"_all_observable_features_" + str(max_feats) + ".pickle") and os.path.exists("parallel_pickles/" +map_name+ "_experiment"+str(experiments-1)+"tree_ids.pickle"):
		print("loading existing pickles...")
		with open("parallel_pickles/"+map_name+"_all_observable_features_" + str(max_feats) + ".pickle","rb") as handle:
			all_possible_detections = pickle.load(handle)
		with open("parallel_pickles/"+map_name+"_clique_feats" + str(max_feats) + ".pickle","rb") as handle:
			clique_features = pickle.load(handle)
		with open("parallel_pickles/" +map_name+ "_experiment"+str(experiments-1)+"tree_ids.pickle","rb") as handle:
			tree_ids = pickle.load(handle)
		with open("parallel_pickles/" +map_name+ "_experiment"+str(experiments-1)+"cone_ids.pickle","rb") as handle:
			cone_ids = pickle.load(handle)
	else:
		print("Finding all the observable features... this might take a while!")
		print("this is experiments: {} and sim_length: {}".format(experiments,sim_length))
		width = 1392; height = 1024; fov = 72 #these are the parameters of the camera we used in carla to get the orb features :)
		camera_offset = [0.3,0,1.7] #this is the translation from the car frame to the camera frame 
		clique_features,all_possible_detections,tree_ids,cone_ids = observable_features_carla(map_name,results_dir,min_feats,max_feats,
			max_dist,is_visual,experiments,sim_length,width,height,fov,exp_gt_traj,camera_offset,miss_detection_probability_function,P_Miss_detection,P_False_detection,sensor_noise_variance)
		print("Found all the features for one experiment... lets check it out")
		print("clique_features: ",clique_features)
		raise OSError 
	### end finding features ###

	### find gt lm persitence and lm centers for performance 
	if os.path.exists("parallel_pickles/"+map_name+ "_experiment"+str(experiments-1)+"gt_lms.pickle") and os.path.exists("parallel_pickles/"+map_name+ "_experiment"+str(experiments-1)+"gt_lm_persistence.pickle"):
		with open("parallel_pickles/"+map_name+ "_experiment"+str(experiments-1)+"gt_lms.pickle","rb") as handle:
			gt_lms = pickle.load(handle)
		with open("parallel_pickles/"+map_name+ "_experiment"+str(experiments-1)+"gt_lm_persistence.pickle","rb") as handle:
			gt_lm_persistence = pickle.load(handle)
	else:
		gt_lms = get_gt_carla_landmarks(results_dir,experiments,sim_length,all_possible_detections) #these are the gt centers of all landmarks 
		with open("parallel_pickles/"+map_name+ "_experiment"+str(experiments-1)+"gt_lms.pickle","wb") as handle:
			pickle.dump(gt_lms,handle,protocol=pickle.HIGHEST_PROTOCOL)
		print("gt_lms: ",gt_lms)
		gt_lm_persistence = get_lm_persistence(results_dir,experiments,sim_length,gt_lms,tree_ids,cone_ids)
		with open("parallel_pickles/"+map_name+ "_experiment"+str(experiments-1)+"gt_lm_persistence.pickle","wb") as handle:
			pickle.dump(gt_lm_persistence,handle,protocol=pickle.HIGHEST_PROTOCOL)
		#gt_lm_persistence contains cliques with too few features 

	clique_features, gt_lm_persistence = check_clique_ids(clique_features,gt_lm_persistence)
	### end finding gt lm data ### 

	print("initializing the clique sim...")
	#logS_T = lambda t: -lambda_u * t
	##BetterTogether Initialization###
	betterTogether_sims = []
	for d in data_association_rates:
		sim_d = clique_simulator(results_dir,clique_features,experiments,sim_length,max_sim_length,negative_supression,P_Miss_detection,P_False_detection,confidence_range,lambda_u,
			detection_threshold,rejection_threshold,tree_ids,cone_ids)
		betterTogether_sims.append(sim_d)
	###BetterTogether Initialization### 

	if betterFaster:
		betterFaster_sims_bad_growth_model = []
		betterFaster_sims_good_growth_model = []
		for d in data_association_rates:
			betterFaster_sim_d = clique_simulator(results_dir,clique_features,experiments,sim_length,max_sim_length,negative_supression,P_Miss_detection,P_False_detection,confidence_range,lambda_u,
				detection_threshold,rejection_threshold,tree_ids,cone_ids)
			betterFaster_sims_bad_growth_model.append(betterFaster_sim_d)
			betterFaster_sim_d = clique_simulator(results_dir,clique_features,experiments,sim_length,max_sim_length,negative_supression,P_Miss_detection,P_False_detection,confidence_range,lambda_u,
				detection_threshold,rejection_threshold,tree_ids,cone_ids)
			betterFaster_sims_good_growth_model.append(betterFaster_sim_d)

	#Init SLAM
	betterTogether_SLAMs = []
	for d in data_association_rates:
		SLAM = fastSLAM2(n_particles,gt_traj[0],init_state_covar,[x for x in gt_lms.keys()]) 
		betterTogether_SLAMs.append(SLAM)

	if betterFaster:
		betterFaster_SLAMs_good_growth_model = []
		betterFaster_SLAMs_bad_growth_model = []
		for d in data_association_rates:
			tuned_SLAM = fastSLAM2(n_particles,gt_traj[0],init_state_covar,[x for x in gt_lms.keys()])
			betterFaster_SLAMs_good_growth_model.append(tuned_SLAM)
			tuned_SLAM = fastSLAM2(n_particles,gt_traj[0],init_state_covar,[x for x in gt_lms.keys()])
			betterFaster_SLAMs_bad_growth_model.append(tuned_SLAM)

	### INIT COMPARISON METHODS ### 

	if multimap:
		numActive_maps = multimap_params["numActive_maps"]
		lmin = multimap_params["lmin"]
		lmax = multimap_params["lmax"]
		pmiss = multimap_params["pmiss"]
		phit = multimap_params["phit"]
		init_weight = multimap_params["init_weight"]
		init_t = multimap_params["init_t"]
		epsilon = multimap_params["epsilon"]

		MM_SLAMs = []; multiMap_sims = []
		for d in data_association_rates:
			MM_SLAM = fastSLAM2(n_particles,gt_traj[0],init_state_covar,[x for x in gt_lms.keys()])
			multiMap = multiMap_simulator(MM_SLAM,init_t,numActive_maps,lmin,lmax,pmiss,phit,init_weight,detection_threshold,
				rejection_threshold,n_particles,epsilon)
			MM_SLAMs.append(MM_SLAM)
			multiMap_sims.append(multiMap)

	if vanilla:
		vanilla_SLAMs = [] 
		for i in range(len(data_association_rates)):
			vanilla_SLAMs.append(fastSLAM2(n_particles,gt_traj[0],init_state_covar,[x for x in gt_lms.keys()]))

	### END INIT COMPARISON METHODS ###

	print("Done initializing... starting the experiments!")

	### LOGGING ###
	flavors = []
	for d in data_association_rates:
		flavors.append("betterTogether" + str(d))
		if vanilla:
			flavors.append("vanilla"+str(d))
	bf_time = []
	if multimap:
		mm_time = []
		for d in data_association_rates:
			flavors.append("multiMap" + str(d))
	if betterFaster:
		tuned_time = []
		for d in data_association_rates:
			flavors.append("betterFaster_bad_growth_model"+str(d))
			flavors.append("betterFaster_good_growth_model"+str(d))

	#possible_detections,flavors,results_dir,sim_length,experiments,gt_traj,gt_lms,gt_lm_persistence,rejection_threshold,detection_threshold,tree_ids,cone_ids
	perf = Performance(all_possible_detections,flavors,results_dir,sim_length,experiments,gt_traj,gt_lms,gt_lm_persistence,rejection_threshold,detection_threshold,tree_ids,cone_ids)
	### END LOGGING ###

	print("Done initializing... starting to run experiments")
	for run in range(experiments):
		print("starting experiment... ",run)
		if np.mod(run,experiments_season) == 0 and run > 0:
			for i,d in enumerate(data_association_rates):
				print("seasonal data missasociation is happening to betterTogether"+str(d))
				seasonal_data_misassociation(betterTogether_SLAMs[i],d)
				if multimap:
					print("seasonal data missasociation is happening to multiMap"+str(d))
					seasonal_data_misassociation(MM_SLAMs[i],d)
					reinit_multimaps(multiMap_sims[i],d)
				if vanilla:
					print("seasonal data missasociation is happening to vanilla"+str(d))
					seasonal_data_misassociation(vanilla_SLAMs[i],d)
		exp_errs = 0
		for i in range(sim_length):
			t = run*sim_length + i 
			percent_done = 100*(t/(experiments*sim_length))

			print("this is t: {}... we are {} percent done!".format(t,np.round(percent_done,2)))
			print("this is experiment: {} sim_length: {} map_name: {}".format(experiments,sim_length,map_name))
			pose = gt_traj[t] #traditionally, this is done using the vehicle odometry 
			#in this case we take a noisy estimate based on the ground truth pose to emulate a GPS measurement (for example)

			detections = [x for x in all_possible_detections[t] if x['clique_id'] in clique_features.keys()]
			#print("these are the possible detections: ",[x['clique_id'] for x in detections])

			betterTogether_detections_t = {}
			for i,d in enumerate(data_association_rates):
				betterTogether_detections = get_realistic_detections(run,results_dir,clique_features,gt_lms,tree_ids,cone_ids,detections,P_False_detection,d)
				#print("betterTogether" + str(d) + " detections: ",[x['clique_id'] for x in betterTogether_detections])
				betterTogether_detections_t[d] = betterTogether_detections

				obsd_cliques = [x['clique_id'] for x in detections]

				SLAM = betterTogether_SLAMs[i]
				sim_i = betterTogether_sims[i]

				SLAM.prediction(pose)
				bf_time0 = time.time()
				persistent_observations = sim_i.update((t,betterTogether_detections,False))
				bf_time.append([t,len(betterTogether_detections),time.time() - bf_time0])
				#print("this is betterTogether calling corrections... this is observations: ",persistent_observations)
				SLAM.correction((t,persistent_observations))

			if betterFaster:
				bad_betterFaster_detections_t = {}
				good_betterFaster_detections_t = {}
				for i,d in enumerate(data_association_rates):
					#print("data_association_rate: ",d)
					good_tuned_SLAM = betterFaster_SLAMs_good_growth_model[i]
					bad_tuned_SLAM = betterFaster_SLAMs_bad_growth_model[i]
					good_tuned_SLAM.prediction(pose)
					bad_tuned_SLAM.prediction(pose)
					good_sim = betterFaster_sims_good_growth_model[i]
					bad_sim = betterFaster_sims_bad_growth_model[i]
					tuned_time0 = time.time()
					good_betterFaster_detections = get_betterFaster_detections(("good",results_dir,clique_features,gt_lms,tree_ids,cone_ids,gt_lm_persistence,data_association_rates[i],good_sim,detections,
						detection_threshold,P_False_detection,t,run))
					#print("betterFaster" + str(d) + " detections - good growth state model: ",len(good_betterFaster_detections))
					#print("good_betterFaster_detections: ",[x['clique_id'] for x in good_betterFaster_detections])
					good_betterFaster_detections_t[d] = good_betterFaster_detections
					persistent_observations = good_sim.update((t,good_betterFaster_detections,True))
					tuned_time.append([t,len(good_betterFaster_detections),time.time()-tuned_time0])
					good_tuned_SLAM.correction((t,persistent_observations))
					tuned_time0 = time.time()
					bad_betterFaster_detections = get_betterFaster_detections(("bad",results_dir,clique_features,gt_lms,tree_ids,cone_ids,gt_lm_persistence,data_association_rates[i],good_sim,detections,
						detection_threshold,P_False_detection,t,run))
					#print("betterFaster" + str(d) + " detections - bad growth state model: ",len(bad_betterFaster_detections))
					#print("bad_betterFaster_detections: ",[x['clique_id'] for x in bad_betterFaster_detections])
					bad_betterFaster_detections_t[d] = bad_betterFaster_detections
					persistent_observations = bad_sim.update((t,bad_betterFaster_detections,True))
					tuned_time.append([t,len(bad_betterFaster_detections),time.time()-tuned_time0])
					#print("this is betterFaster calling corrections... this is observations: ",persistent_observations)
					bad_tuned_SLAM.correction((t,persistent_observations))

			if multimap:
				multimap_detections_t = {}
				for i,d in enumerate(data_association_rates):
					#print("data_association_rate: ",d)
					MM_SLAM = MM_SLAMs[i]; multiMap = multiMap_sims[i]
					detections_d = get_realistic_detections(run,results_dir,clique_features,gt_lms,tree_ids,cone_ids,detections,P_False_detection,d)
					multimap_detections_t[d] = detections_d
					#print("multimap detections: ",[x['clique_id'] for x in detections_d])
					MM_SLAM.prediction(pose)
					mm_time0 = time.time()

					multimap_observations = multiMap.update((t,detections_d))
					mm_time.append([t,len(detections),time.time() - mm_time0])
					#print("this is multiMap calling corrections... this is observations: ",multimap_observations)
					MM_SLAM.correction((t,multimap_observations))

			if vanilla:
				vanilla_detections_t = {}
				for i,v in enumerate(vanilla_SLAMs):
					#print("data_association_rate: ",data_association_rates[i])
					v.prediction(pose)
					#run,results_dir,gt_lms,tree_ids,cone_ids,detections,p_false_detection,data_association_rate
					vanilla_detections = get_realistic_detections(run,results_dir,clique_features,gt_lms,tree_ids,cone_ids,detections,P_False_detection,data_association_rates[i])
					vanilla_detections_t[data_association_rates[i]] = vanilla_detections
					#print("this is vanilla" +str(data_association_rates[i]) +"... this is observations: ",[x['clique_id'] for x in vanilla_detections])
					v.correction((t,vanilla_detections))

			#compute performance 
			if betterFaster:
				bad_gmodel_perf_t = {}
				good_gmodel_perf_t = {}
				for i,d in enumerate(data_association_rates):
					#bad growth model
					#print("data_association_rate: ",d)
					betterFaster_sim = betterFaster_sims_bad_growth_model[i]
					tuned_SLAM = betterFaster_SLAMs_bad_growth_model[i]
					sim_type = "betterFaster_bad_growth_model"+str(d)
					#run,t,detections,sim_detections,sim,SLAM,sim_type,tree_ids,cone_ids
					bad_gmodel_perf_t[d] = perf.compute_performance((run,t,detections,bad_betterFaster_detections_t[d],betterFaster_sim,tuned_SLAM,sim_type,tree_ids,cone_ids))
					#good growth model 
					betterFaster_sim = betterFaster_sims_good_growth_model[i]
					tuned_SLAM = betterFaster_SLAMs_good_growth_model[i]
					sim_type = "betterFaster_good_growth_model"+str(d)
					good_gmodel_perf_t[d] = perf.compute_performance((run,t,detections,good_betterFaster_detections_t[d],betterFaster_sim,tuned_SLAM,sim_type,tree_ids,cone_ids))

			betterTogether_perf_t = {}
			for i,d in enumerate(data_association_rates):
				sim = betterTogether_sims[i]
				SLAM = betterTogether_SLAMs[i]
				betterTogether_perf_t[d] = perf.compute_performance(run,t,detections,betterTogether_detections_t[d],sim,SLAM,"betterTogether" + str(d),tree_ids,cone_ids)

			if multimap:
				multiMap_perf_t = {}
				for i,d in enumerate(data_association_rates):
					multiMap = multiMap_sims[i]; MM_SLAM = MM_SLAMs[i]
					multiMap_perf_t[d] = perf.compute_performance(run,t,detections,multimap_detections_t[d],multiMap,MM_SLAM,"multiMap" + str(d),tree_ids,cone_ids)

			if vanilla:
				vanilla_perf_t = {}
				for i,d in enumerate(data_association_rates):
					vanilla_perf_t[d] = perf.compute_performance(run,t,detections,vanilla_detections_t[d],[],vanilla_SLAMs[i],"vanilla" + str(d),tree_ids,cone_ids)
			#end compute performance 
		print("Finished experiment {}!".format(run))

		if not skip_writing_files:
			print("writing output files...")
			if not os.path.exists(str(experiments) + "results"):
				os.mkdir(str(experiments) + "results")
			param_sweep_type = []
			for i,d in enumerate(data_association_rates):
				sim_d = betterTogether_sims[i]
				write_files(map_name,experiments,"betterTogether"+str(d),sim_d,perf,run,param_sweep_type)
			if betterFaster:
				for i,d in enumerate(data_association_rates):
					betterFaster_sim = betterFaster_sims_good_growth_model[i]
					write_files(map_name,experiments,"betterFaster_good_growth_model"+str(d),betterFaster_sim,perf,run,param_sweep_type)
					betterFaster_sim = betterFaster_sims_bad_growth_model[i]
					write_files(map_name,experiments,"betterFaster_bad_growth_model"+str(d),betterFaster_sim,perf,run,param_sweep_type)
			if multimap:
				for i,d in enumerate(data_association_rates):
					write_files(map_name,experiments,"multiMap"+str(d),multiMap,perf,run,param_sweep_type)
			if vanilla:
				for d in data_association_rates:
					write_files(map_name,experiments,"vanilla" + str(d), [], perf, run,param_sweep_type)

if __name__ == "__main__":
	### SIM PARAMETERS ###    
	P_Miss_detection = 0.1
	P_False_detection = 0.05

	### SIM PARAMETERS ### 
	#lambda_u - .001, .005, .01
	lambda_u = 1/350.0 #.0029


	detection_threshold = 0.9 #0.5,0.75,0.9
	rejection_threshold = 0.5 #0.25,0.5,0.75

	experiments = 12 #max = 12 

	experiments_season = 3 #experiments for seasons

	sim_length = 500 #This is the number of timesteps you want to run (max = 500)

	min_feats = 3
	max_feats = 100 #max = 1000 (unless you want to run observable fatures carla again)

	sensor_noise_variance = 0.1 #0.05,0.1,0.25

	confidence_range = 100 

	#self.miss_detection_probability_function = lambda d: 1 - np.exp(-1.0/self.confidence_range * d)
	miss_detection_probability_function = lambda d: -2*10**(-6)*d**2 + .0025*d 

	### END SIM PARAMETERS ### 

	negative_supression = True

	is_visual = True #select if you want to use visual-inertial SLAM or use lidar-inertial SLAM
	#TO DO: implement lidar-inertial SLAM version 

	#run comparisons to BetterTogether
	betterFaster = True 
	multimap = True
	vanilla = True

	skip_writing_files = False #Change to True if you dont want to save the results 

	#Multimap params
	multimap_params = {}
	multimap_params["numActive_maps"] = 10
	multimap_params["lmin"] = [.4, -2.9]; 
	multimap_params["lmax"] = [1.28, 4.59];
	multimap_params["pmiss"] = .1; 
	multimap_params["phit"] = .9;
	multimap_params["init_weight"] = 0.3
	multimap_params["init_t"] = sim_length*.25 
	multimap_params["epsilon"] = .001

	#map_name = "empty_map_results"
	map_name = "town01_results"
	results_dir = "/home/kristen/BetterFaster2.0/run_carla_experiments/"
	run_betterFaster(P_Miss_detection,P_False_detection,lambda_u,detection_threshold,rejection_threshold,experiments,experiments_season,
		sim_length,sensor_noise_variance,confidence_range,miss_detection_probability_function,min_feats,max_feats,results_dir,map_name,
		multimap_params)