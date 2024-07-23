import numpy as np
import scipy  
import os
import threading 
from scipy.stats import multivariate_normal
from betterFaster.SLAMCore.ekf_filter_utils import wrap2pi,norm_pdf_multivariate 
import concurrent.futures 
from numpy.linalg import LinAlgError

def all_arrays_equal(array_list):
	if not array_list:
		return True
	first_array = array_list[0] 
	for array in array_list[1:]:
		if not np.array_equal(first_array, array):
			return False
	return True

class ekf_filter(): 
	def __init__(self,mu,sigma,Hfun,hfun,Qt): 
		self.mu = mu
		self.Sigma = sigma 
		self.Hfun = Hfun
		self.hfun = hfun
		self.p = 0
		self.Qt =  Qt
	'''
	def update(self,z,x):
		#init_mu = [x for x in self.mu]
		zhat = self.hfun(self.mu[0],self.mu[1],x)
		H = self.Hfun(self.mu[0],self.mu[1],x,zhat)
		Q = H @ self.Sigma @ H.T + self.Qt
		K = self.Sigma @ H.T @ np.linalg.inv(Q)
		v = z - zhat #innovation
		self.mu += K @ v 
		init_sigma = self.Sigma 
		self.Sigma = (np.identity(2) - K @ H) @ self.Sigma 
		
		if not np.isnan(norm_pdf_multivariate(z,zhat,Q)): 
			self.p = norm_pdf_multivariate(z,zhat,Q)
		else: 
			print("p is nan? ") 
			print("this is x:",x) 
			print("init_sigma: ",init_sigma)
			print("self.Sigma: ",self.Sigma) 
			print("innovation: ",v) 
			print("K@v: ", K @ v) 
			print("this is z: {}, zhat: {} and Q: {}".format(z,zhat,Q)) 
			raise OSError 
	'''
	def update(self, z, xhat):
		# Compute the predicted measurement zhat using hfun
		zhat = self.hfun(self.mu[0],self.mu[1],xhat)

		# Compute the Jacobian matrix H using Hfun
		H = self.Hfun(self.mu[0],self.mu[1],xhat,zhat)

		# Compute the innovation (measurement residual)
		innovation = z - zhat

		# Compute the innovation covariance S
		S = H @ self.Sigma @ H.T + self.Qt

		# Compute the Kalman gain K
		K = self.Sigma @ H.T @ np.linalg.inv(S)

		# Update the state estimate
		self.mu = self.mu + K @ innovation

		# Update the covariance estimate
		I = np.eye(self.Sigma.shape[0])  # Identity matrix
		self.Sigma = (I - K @ H) @ self.Sigma

		# Compute the likelihood of the innovation
		#self.p = multivariate_normal.pdf(innovation, mean=np.zeros(len(innovation)), cov=S) 
		self.p = norm_pdf_multivariate(z,zhat,S) 

		if np.isnan(self.p): 
			print("p is nan? ") 
			print("this is x:",xhat)  
			print("self.Sigma: ",self.Sigma) 
			print("innovation: ",innovation) 
			print("K@v: ", K @ innovation) 
			print("this is z: {}, zhat: {} and S: {}".format(z,zhat,S))  
			raise OSError 
		
class fast_slam_landmark(): 
	def __init__(self,observed,mu,sigma,Hfun,hfun,Qt,lm_id):
		self.EKF =  ekf_filter(mu,sigma,Hfun,hfun,Qt)
		self.isobserved = observed
		self.lm_id = lm_id 

class fast_slam_particle():
	def __init__(self,mu,sigma,n,landmark_ids,Hfun,hfun,Qt): 
		num_landmarks = len(landmark_ids)
		if len(mu) != 3:
			raise OSError 
		#print("mu: {}, sigma: {}".format(mu,sigma))
		self.pose = np.random.multivariate_normal(mu,sigma)
		if len(self.pose) != 3:
			#print("self.pose: ",self.pose)
			raise OSError
		self.weight = 1/n 
		self.landmark = []
		for l in range(num_landmarks):
			lm_id = int(landmark_ids[l])
			lm = fast_slam_landmark(False,np.zeros((2,1)),np.zeros((2,2)),Hfun,hfun,Qt,lm_id)  
			self.landmark.append(lm)

class fastSLAM2():
	#n_particles,localization_covariance,observed_clique_ids
	def __init__(self,n_particles,gt_init_state_mu,localization_covar,lm_ids):
		self.n = n_particles 
		self.mu = np.random.multivariate_normal(gt_init_state_mu,localization_covar) 
		#self.Sigma = np.ones((3,3))*0.05
		self.Sigma = localization_covar
		betas    = [1.25, 2, 0.175]
		betas    = [1.25, 2, 0.175]
		self.Qt = np.dot(betas[0],np.array([[betas[1]**2,np.random.rand()],[np.random.rand(),betas[2]**2]]))
		#print("self.Qt.shape: ",self.Qt.shape)
		#input("Press Enter to continue") 
		#self.Qt = betas 
		#initializing the particles
		self.particles = [fast_slam_particle(self.mu,self.Sigma,self.n,lm_ids,self.Hfun,self.hfun,self.Qt) for x in range(self.n)] 

	def hfun(self,lm_x,lm_y,x):
		return np.array([np.sqrt((lm_y-x[1])**2 + (lm_x-x[0])**2),np.arctan2(self.mu[1] - x[1],self.mu[0] - x[0])-x[2]])
	
	def Hfun(self,lm_x,lm_y,x,z_hat):
		return np.array([[(lm_x - x[0])/z_hat[0], (lm_y-x[1])/z_hat[0]],[(x[1] - lm_y)/z_hat[0]**2, (lm_x-x[0])/z_hat[0]**2]])
	
	def h_inv(self,z,x):
		return np.array([x[0] + z[0]*np.cos(x[2]+z[1]),x[1]+z[0]*np.sin(x[2]+z[1])])

	#this is the original prediction function 
	def prediction(self,pose):
		#print("in prediction... this is pose:",pose)
		for k in range(self.n):
			self.particles[k].pose = np.random.multivariate_normal(pose,self.Sigma)
		best_idx = np.argmax([self.particles[k].weight for k in range(self.n)])
		#change deg 2 rad 
		#print("best pose estimate: ",self.particles[best_idx].pose)
		self.particles[best_idx].pose[2] = np.deg2rad(self.particles[best_idx].pose[2]) 
		#print("returning this: ",self.particles[best_idx].pose)
		return self.particles[best_idx].pose

	'''
	def prediction(self,pose):
		#parallelized prediction 
		with concurrent.futures.ThreadPoolExecutor() as executor:
			executor.map(lambda k: self.parallel_prediction_helper(pose,k), range(self.n))
		best_idx = np.argmax([self.particles[k].weight for k in range(self.n)])
		#print("self.partciles[best_idx].pose: ",self.particles[best_idx].pose) # <- this is degrees 
		#self.particles[best_idx].pose[2] = wrap2pi(self.particles[best_idx].pose[2]*(np.pi/180))
		self.particles[best_idx].pose[2] = wrap2pi(self.particles[best_idx].pose[2]*(np.pi/180))
		#print("self.partciles[best_idx].pose: ",self.particles[best_idx].pose) 
		#input("Press Enter To Continue ")
		return self.particles[best_idx].pose
	
	def parallel_prediction_helper(self,pose,k): 
		self.particles[k].pose = np.random.multivariate_normal(pose,self.Sigma) 
	'''

	#this is the original correction function 
	def correction(self,t,z):
		print("inside fast slam correction ... ")
		debug_ranges = {}
		debug_bearings = {}

		for k in range(self.n):
			particles = self.particles[k] 
			particles_landmark = particles.landmark 
			#print("in fastSLAM... this is z: ",z)
			if not "None" in str(type(z)):
				for observation in z:
					#print("z[j]:",z[j])
					range_val = observation.get('range')
					bearing_val = observation.get('bearing') #for carla this is in degrees 
					bearing_rad = bearing_val * (np.pi/180) 
					id_ = observation.get('clique_id')   
					#DEBUGGING 
					if id_ not in debug_ranges.keys() or id_ not in debug_bearings.keys(): 
						debug_ranges[id_] = [] 
						debug_bearings[id_] = []
					debug_ranges[id_].append(range_val) 
					debug_bearings[id_].append(bearing_val)  
					#END DEBUGGING 
					z_t = np.array([range_val, bearing_rad])  
					if not np.isnan(bearing_val) and not np.isnan(range_val):
						#print("observing landmark: ",id_)
						if id_ in [x.lm_id for x in self.particles[k].landmark]: 
							idx_lm = next(i for i, lm in enumerate(particles_landmark) if lm.lm_id == id_) 
							if not self.particles[k].landmark[idx_lm].isobserved:
								#print("this landmark was not observed yet ... initting") 
								mean_init = self.h_inv(z_t,self.particles[k].pose)
								H = self.Hfun(mean_init[0],mean_init[1],self.particles[k].pose,z_t)
								H_inv = np.identity(2) / H 
								cov_init = H_inv @ self.Qt @ H_inv.T
								#print("mean_init: ",mean_init)
								self.particles[k].landmark[idx_lm].EKF.mu = mean_init
								self.particles[k].landmark[idx_lm].EKF.Sigma = cov_init
								self.particles[k].landmark[idx_lm].isobserved = True
								#print("particle k:{} was observed".format(k))
							else:
								#print("callling update ...")
								self.particles[k].landmark[idx_lm].EKF.update(z_t,self.particles[k].pose)
								#print("this is p: ",self.particles[k].landmark[idx_lm].EKF.p) 
								if np.isnan(self.particles[k].landmark[idx_lm].EKF.p):
									raise OSError 
								self.particles[k].weight = self.particles[k].weight * self.particles[k].landmark[idx_lm].EKF.p
			else:
				print("observations is none type....")
				raise OSError 

		landmark_centers = {} 
		for particle in self.particles:
			landmarks = particle.landmark 
			landmark_ids = [x.lm_id for x in landmarks]  
			#print("landmark_ids: ",landmark_ids) 
			for id_ in landmark_ids: 
				if id_ not in landmark_centers.keys(): 
					landmark_centers[id_] = [] 
				landmark_centers[id_].append([x.EKF.mu for x in landmarks if x.lm_id == id_][0])

		for id_ in landmark_centers.keys(): 
			if not np.all(landmark_centers[id_][0] == 0): 
				if all_arrays_equal(landmark_centers[id_]): 
					print("WARNING all landmark centers are the same?") 
					print("landmark centers: ",landmark_centers[id_]) 
					raise OSError 
					
		'''
		for id_ in debug_ranges.keys(): 
			print()
			print("this is id_:",id_) 
			print("this is the mean observed range of this id_: ",np.mean(debug_ranges[id_])) 
			print("this is the mean observed bearing of this id_ ",np.mean(debug_bearings[id_])) 
			print()
		'''

		#Update weights 
		if np.any(np.isnan([x.weight for x in self.particles])):
			raise OSError 
		
		weight_sum = np.sum([x.weight for x in self.particles])
		print("weight sum: ",weight_sum)
		for k in range(self.n): 
			self.particles[k].weight = self.particles[k].weight/weight_sum  
		weight_sum = np.sum([x.weight for x in self.particles]) 
		neff = weight_sum**(-2) 
		print("neff: ",neff) 
		if neff < self.n /2 or np.isnan(neff): 
			self.resample()

	'''
	def correction(self, t, z):
		with concurrent.futures.ThreadPoolExecutor() as executor:
			executor.map(lambda k: self.parallel_correction_helper(z, k), range(self.n))

		# Update weights
		weight_sum = np.sum([x.weight for x in self.particles])
		for k in range(self.n):
			self.particles[k].weight /= weight_sum

		weight_sum = np.sum([x.weight for x in self.particles])
		neff = weight_sum ** (-2)
		if neff < self.n / 2:
			self.resample()
			
	def parallel_correction_helper(self, z, k):
		#This is the ChatGPT re-write
		#gt_lms_first_column = self.gt_lms[:, 0]
		particles = self.particles[k]
		particles_landmark = particles.landmark

		for observation in z:
			#print("z[j]:",z[j])
			range_val = observation.get('range')
			bearing_val = observation.get('bearing') #for carla this is in degrees 
			#print("range_val: {}, bearing_val: {}".format(range_val,bearing_val))
			#print("bearing val: ",bearing_val) 
			bearing_rad = bearing_val * (np.pi/180) 
			id_ = observation.get('clique_id')  
			z_t = np.array([range_val, bearing_rad])  
			if not np.isnan(bearing_val) and not np.isnan(range_val):
				#print("observing landmark: ",id_)
				if id_ in [x.lm_id for x in self.particles[k].landmark]: 
					idx_lm = next(i for i, lm in enumerate(particles_landmark) if lm.lm_id == id_) 
					if not self.particles[k].landmark[idx_lm].isobserved:
						mean_init = self.h_inv(z_t,self.particles[k].pose)
						H = self.Hfun(mean_init[0],mean_init[1],self.particles[k].pose,z_t)
						H_inv = np.identity(2) / H 
						cov_init = H_inv @ self.Qt @ H_inv.T
						#print("mean_init: ",mean_init)
						self.particles[k].landmark[idx_lm].EKF.mu = mean_init
						self.particles[k].landmark[idx_lm].EKF.Sigma = cov_init
						self.particles[k].landmark[idx_lm].isobserved = True
						#print("particle k:{} was observed".format(k))
					else:
						self.particles[k].landmark[idx_lm].EKF.update(z_t,self.particles[k].pose)
						self.particles[k].weight = self.particles[k].weight * self.particles[k].landmark[idx_lm].EKF.p 
	''' 

	def resample(self):
		print("resampling...") 
		W = np.cumsum([x.weight for x in self.particles]) 
		print("[x.weight for x in self.particles]: ",[x.weight for x in self.particles]) 
		print("W:",W) 
		r = np.random.rand()/self.n 
		j = 1 
		for i in range(self.n): 
			u = r + (i-1)/self.n 
			while u > W[j]: 
				j += 1 
			self.particles[i] = self.particles[j] 
			self.particles[i].weight = 1/self.n 

	def reinit_EKF(self,id_,detections_t,t,experiment):
		for k in range(self.n):
			idx_lm = [i for i,x in enumerate(self.particles[k].landmark) if x.lm_id==id_][0]
			self.particles[k].landmark[idx_lm].isobserved = False 

		if len(detections_t) >0:
			self.correction(detections_t,t,experiment)