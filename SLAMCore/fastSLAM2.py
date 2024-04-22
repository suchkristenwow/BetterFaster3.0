import numpy as np
import scipy  
import os

from ekf_filter_utils import *

class ekf_filter(): 
	def __init__(self,mu,sigma,Hfun,hfun,Qt): 
		self.mu = mu
		self.Sigma = sigma 
		self.Hfun = Hfun
		self.hfun = hfun
		self.p = 0
		self.Qt =  Qt
		self.debug_count = 0
		self.feature_cache = {}  

	def update(self,z,x):
		init_mu = [x for x in self.mu]
		zhat = self.hfun(self.mu[0],self.mu[1],x)
		H = self.Hfun(self.mu[0],self.mu[1],x,zhat)
		Q = H @ self.Sigma @ H.T + self.Qt
		K = self.Sigma @ H.T @ np.linalg.inv(Q)
		v = z - zhat #innovation
		self.mu += K @ v 
		self.Sigma = (np.identity(2) - K @ H) @ self.Sigma 
		self.p = norm_pdf_multivariate(z,zhat,Q)

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
	def __init__(self,n_particles,init_state_covar,lm_ids):
		self.n = n_particles 
		self.mu = np.zeros((3,))
		self.Sigma = init_state_covar
		betas    = [1.25, 2, 0.175]
		self.Qt = np.dot(betas[0],np.array([[betas[1]**2,np.random.rand()],[np.random.rand(),betas[2]**2]]))
		#initializing the particles
		self.particles = [fast_slam_particle(self.mu,self.Sigma,self.n,lm_ids,self.Hfun,self.hfun,self.Qt) for x in range(self.n)] 

	def hfun(self,lm_x,lm_y,x):
		return np.array([np.sqrt((lm_y-x[1])**2 + (lm_x-x[0])**2),np.arctan2(self.mu[1] - x[1],self.mu[0] - x[0])-x[2]])
	
	def Hfun(self,lm_x,lm_y,x,z_hat):
		return np.array([[(lm_x - x[0])/z_hat[0], (lm_y-x[1])/z_hat[0]],[(x[1] - lm_y)/z_hat[0]**2, (lm_x-x[0])/z_hat[0]**2]])
	
	def h_inv(self,z,x):
		return np.array([x[0] + z[0]*np.cos(x[2]+z[1]),x[1]+z[0]*np.sin(x[2]+z[1])])

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
	
	def correction(self,t,z):
		#t = args[0]; z = args[1]
		#print("observations_t: ",z)
		for k in range(self.n):
			#print("in fastSLAM... this is z: ",z)
			if not "None" in str(type(z)):
				for j in range(len(z)):
					#print("z[j]:",z[j])
					z_t = [z[j]['range'], z[j]['bearing']]
					if not np.isnan(z[j]['bearing']) and not np.isnan(z[j]['range']):
						id_ = z[j]['clique_id']
						#print("observing landmark: ",id_)
						if id_ in [x.lm_id for x in self.particles[k].landmark]:
							idx_lm = [i for i,x in enumerate(self.particles[k].landmark) if x.lm_id == id_][0]
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
			else:
				print("observations is none type....")
				raise OSError
		#Update weights 
		weight_sum = np.sum([x.weight for x in self.particles])
		for k in range(self.n): 
			self.particles[k].weight = self.particles[k].weight/weight_sum 
		weight_sum = np.sum([x.weight for x in self.particles])
		neff = weight_sum**(-2) 
		if neff < self.n /2: 
			self.resample()

	def resample(self):
		W = np.cumsum([x.weight for x in self.particles])
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