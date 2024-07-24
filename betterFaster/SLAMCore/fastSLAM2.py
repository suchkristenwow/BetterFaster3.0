import numpy as np 
import concurrent.futures  

class fast_slam_landmark(): 
	def __init__(self,lm_id,mu,sigma):
		self.isobserved = False
		self.lm_id = lm_id 
		self.mu = mu 
		self.sigma = sigma 

class fast_slam_particle(): 
	def __init__(self,pose,weight,lm_ids): 
		self.pose = pose 
		self.weight = weight 
		self.landmarks = []
		for id_ in lm_ids:
			mu = np.zeros((2,)); sigma = np.zeros((2,2)) 
			self.landmarks.append(fast_slam_landmark(id_,mu,sigma)) 

class fastSLAM2():
	def __init__(self,init_pose,localization_covariance,n_particles,lm_ids,Q_params):  
		self.localization_covariance = localization_covariance 
		self.n = n_particles 
		self.particles = [fast_slam_particle(init_pose,0,lm_ids) for _ in range(n_particles)]
		self.Q = np.diagflat(np.array([Q_params[0], Q_params[1]])) ** 2

	def parallel_prediction_helper(self,k,pose):
		self.particles[k].pose = np.random.multivariate_normal(pose,self.localization_covariance) 

	def prediction(self,pose): 
		with concurrent.futures.ThreadPoolExecutor() as executor:
			executor.map(lambda k: self.parallel_prediction_helper(k,pose), range(self.n))
		particle_weights = [x.weight for x in self.particles] 
		best_particle_idx = np.argmax(particle_weights)
		return self.particles[best_particle_idx].pose 

	def compute_expected_landmark_state(self,particle_idx,measurement):
		robot_x = self.particles[particle_idx].pose[0] 
		robot_y = self.particles[particle_idx].pose[1] 
		robot_heading = self.particles[particle_idx].pose[2]
		x = robot_x + measurement[0] + np.cos(measurement[1] + robot_heading) 
		y = robot_y + measurement[0] + np.sin(measurement[1] + robot_heading) 
		return [x,y] 

	def compute_landmark_jacobian(self,particle_idx,lm_idx): 
		delta_x = self.particles[particle_idx].landmarks[lm_idx].mu[0] - self.particles[particle_idx].pose[0] 		
		delta_y = self.particles[particle_idx].landmarks[lm_idx].mu[1] - self.particles[particle_idx].pose[1] 
		q = delta_x**2 + delta_y**2 
		H_1 = np.array([delta_x/np.sqrt(q), delta_y/np.sqrt(q)])
		H_2 = np.array([-delta_y/q, delta_x/q])
		H_m = np.array([H_1, H_2])
		return H_m
	
	def initialize_landmark(self,particle_idx,lm_idx,measurement): 
		#compute expected landmark state 
		self.particles[particle_idx].landmarks[lm_idx].mu = self.compute_expected_landmark_state(particle_idx,measurement)
 		#compute Jacobian 
		H_m = self.compute_landmark_jacobian(particle_idx,lm_idx) 
		#update landmark covariance
		H_inverse = np.linalg.inv(H_m) 
		self.particles[particle_idx].landmarks[lm_idx].sigma = H_inverse.dot(self.Q).dot(H_inverse.T) 
		#mark landmark as observed 
		self.particles[particle_idx].landmarks[lm_idx].isobserved = True 
		#update weight 
		self.particles[particle_idx].weight = 1/self.n 

	def compute_expected_measurement(self,particle_idx,lm_idx): 
		delta_x = self.particles[particle_idx].landmarks[lm_idx].mu[0] - self.particles[particle_idx].pose[0] 
		delta_y = self.particles[particle_idx].landmarks[lm_idx].mu[1] - self.particles[particle_idx].pose[1] 

		q = delta_x ** 2 + delta_y ** 2

		range = np.sqrt(q)
		bearing = np.arctan2(delta_y, delta_x) - self.particles[particle_idx].pose[2] 

		return range, bearing

	def landmark_update(self,particle_idx,lm_idx,measurement):
		# Compute expected measurement
		range_expected, bearing_expected =\
			self.compute_expected_measurement(particle_idx, lm_idx)

		# Get Jacobian wrt landmark state
		H_m = self.compute_landmark_jacobian(particle_idx, lm_idx)

		# Compute Kalman gain
		Q = H_m.dot(self.particles[particle_idx].landmarks[lm_idx].sigma).dot(H_m.T) + self.Q
		#K = self.particles[particle_idx].landmarks[lm_idx].sigma.dot(H_m.T).dot(np.linalg.inv(Q))
		K = self.particles[particle_idx].landmarks[lm_idx].sigma.dot(H_m.T).dot(np.linalg.inv(Q)) 

		# Update mean
		difference = np.array([[measurement[0] - range_expected],
								[measurement[1] - bearing_expected]])
		innovation = K.dot(difference)
		self.particles[particle_idx].landmarks[lm_idx].mu += innovation.T[0]

		# Update covariance
		self.particles[particle_idx].landmarks[lm_idx].sigma =\
			(np.identity(2) - K.dot(H_m)).dot(self.particles[particle_idx].landmarks[lm_idx].sigma)

		# Importance factor
		self.particles[particle_idx].weight = np.linalg.det(2 * np.pi * Q) ** (-0.5) *\
			np.exp(-0.5 * difference.T.dot(np.linalg.inv(Q)).
					dot(difference))[0, 0]
		
	def normalize_weights(self):  
		sum = 0 
		for particle in self.particles: 
			sum += particle.weight 
	
		#sum = np.cumsum([p.weight for p in self.particles]) 
			
		# If sum is too small, equally assign weights to all particles
		if sum < 1e-10:
			for particle in self.particles:
				particle.weight = 1.0 / self.n
			return

		for particle in self.particles:
			particle.weight /= sum

	def correction(self,z): 
		for k in range(self.n): 
			for observation in z: 
				obs_range = observation.get('range') 
				obs_bearing = observation.get('bearing') 
				obs_id = observation.get('clique_id') 
				z_t = [obs_range,obs_bearing] 
				#check if the landmark has been observed yet or not 
				particle_landmarks = self.particles[k].landmarks 
				#print("[x.lm_id for x in particle_landmarks]: ", [x.lm_id for x in particle_landmarks])
				#print("observation: ",observation)  
				for i,lm in enumerate(particle_landmarks):
					if lm.lm_id == obs_id:
						lm_idx = i 
						break 
				#lm_idx = next(i for i,lm in enumerate(self.particles[k].landmarks) if lm.lm_id == obs_id) 
				if not self.particles[k].landmarks[lm_idx].isobserved:
					#init the landmark 
					self.initialize_landmark(k,lm_idx,z_t) 
				else:
					#update the landmark 
					self.landmark_update(k,lm_idx,z_t)  

			#Normalize weights 
			self.normalize_weights() 
	
	def reinit_EKFs(self,reinit_ids): 
		for k in range(self.n): 
			for id_ in reinit_ids:
				idx = np.where([x.lm_id for x in self.particles[k].landmarks] == id_)[0][0]
				#print("lms: ",[x.lm_id for x in self.particles[k].landmarks] )
				#print("id_: ",id_) 
				#print("idx: ",idx) 
				self.particles[k].landmarks[idx].mu = np.zeros((2,)) 
				self.particles[k].landmarks[idx].sigma = np.zeros((2,2))  
				self.particles[k].landmarks[idx].isobserved = False  