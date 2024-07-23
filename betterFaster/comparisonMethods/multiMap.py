import numpy as np 

class multiMap():
    def __init__(self,params,MM_SLAM): 
        self.STM 				= [] #this is an n by 3 matrix where each row is [id,x,y]
        self.LTM 				= []
        self.STM_logLikely 		= np.empty((len(MM_SLAM.particles),2)) #this is an n by 2 matrix where the 0 column are the ids and the 1 column is the log likely
        self.LTM_logLikely 		= np.empty((len(MM_SLAM.particles),2))
        self.MM_SLAM 			= MM_SLAM 
        self.activeMaps 		= [] 
        self.pendingMaps    	= {}
        self.init_t         	= params["multiMap"]["init_t"]
        self.numActiveMaps  	= params["multiMap"]["numActive_maps"]
        self.lmin 				= params["multiMap"]["lmin"] 
        self.lmax 				= params["multiMap"]["lmax"]
        self.pmiss				= params["multiMap"]["pmiss"]
        self.phit 				= params["multiMap"]["phit"]
        self.init_weight        = params["multiMap"]["init_weight"]
        self.transfer_threshold 	= params["multiMap"]["transfer_threshold"]
        self.rejection_threshold 	= params["multiMap"]["rejection_threshold"] 
        self.n_particles			= params["multiMap"]["n_particles"] 
        self.epsilon				= params["multiMap"]["epsilon"] 
    
    def convert_detections(self,raw_detections): 
        ranges = {} ; bearings = {} 
        for el in raw_detections:
            if el['detection']:
                if el['clique_id'] not in ranges.keys():
                    ranges[el['clique_id']] = el['range']
                    bearings[el['clique_id']] = el['observation_angle']

        detections = [] 
        for lm in ranges.keys():
            mean_range = np.mean(ranges[lm])
            mean_bearing = np.mean(bearings[lm])
            detx = {} 
            detx['clique_id'] = lm 
            detx['range'] = mean_range
            detx['observation_angle'] = mean_bearing
            detections.append(detx)
        return detections 
    
    def update(self,args):
        #update function uses detections to update the long and short term memories 
        #multimap doesnt take into account the indvidual features so I average the detections
        t = args[0]; raw_detections = args[1]

        #average out the detections since this method isnt feature-based 
        detections = self.convert_detections(raw_detections)

        best_particle_idx = np.argmax([self.MM_SLAM.particles[k].weight for k in range(len(self.MM_SLAM.particles))])
        pose = self.MM_SLAM.particles[best_particle_idx].pose
        all_poses = np.array([self.MM_SLAM.particles[k].pose for k in range(len(self.MM_SLAM.particles))])
        tmp = all_poses.T
        pose_covar = np.cov(tmp)
        pose_covar = np.reshape(pose_covar,(3,3))

        if isinstance(self.LTM,list):
            self.LTM = np.array(self.LTM)
            
        for el in detections: 
            #print("this is detection: ",el)
            id_ = el['clique_id']
            if t < self.init_t:
                #add these detections to the LTM
                self.add_detection_to_LTM(el,pose)
            else:
                if len(self.LTM) == 0 or len(self.LTM.shape) <= 1:
                    self.add_detection_to_STM(el,pose)
                else:
                    if id_ not in self.LTM[:,0]:
                        self.add_detection_to_STM(el,pose)

        self.update_STM_logLikely(detections,pose,t)
        self.update_LTM_logLikely(detections,pose,t)

        #all voxels in the STM with log odds value above the threshold are transferred to the LTM
        STM_purge_ids = []
        for i in range(len(self.STM)):
            if isinstance(self.STM,list):
                self.STM = np.array(self.STM)
            if len(self.STM.shape) > 1:
                if len(self.STM_logLikely.shape) > 1:
                    log_likely_idx = [k for k,x in enumerate(self.STM_logLikely[:,0]) if x == self.STM[i,0]]
                else:
                    if self.STM_logLikely[0] == self.STM[i,0]:
                        log_likely_idx = 0 
                    else:
                        print("self.STM:",self.STM)
                        print("self.STM_logLikely:",self.STM_logLikely)
                        raise OSError
            else:
                if len(self.STM_logLikely.shape) > 1:
                    log_likely_idx = [k for k,x in enumerate(self.STM_logLikely[:,0]) if x ==  self.STM[0]]
                else:
                    #print("self.STM:",self.STM)
                    #print("self.STM_logLikely:",self.STM_logLikely)
                    if self.STM_logLikely[0] == self.STM[0]:
                        log_likely_idx = 0
                    else:
                        print("self.STM:",self.STM)
                        print("self.STM_logLikely:",self.STM_logLikely)
                        raise OSError
                    
            #print("logLikely_idx:",log_likely_idx)
            #print("self.STM_logLikely.shape: ",self.STM_logLikely.shape)
            if isinstance(self.STM,list):
                self.STM = np.array(self.STM)
            if len(self.STM_logLikely.shape) > 1:	
                #print("self.STM_logLikely[log_likely_idx,1]:",self.STM_logLikely[log_likely_idx,1][0])
                #print("self.transfer_threshold: ",self.transfer_threshold)
                tmp = self.STM_logLikely[log_likely_idx,1]
                if isinstance(tmp,np.ndarray):
                    tmp = tmp[0]
                if  tmp > self.transfer_threshold:
                    self.LTM = np.vstack([self.LTM,self.STM[i,:]])
                    if len(self.STM.shape) > 1:
                        STM_purge_ids.append(self.STM[i,0])
                    else:
                        STM_purge_ids.append(self.STM[0])
                    self.new_config()
                    self.update_LTM_logLikely(detections,pose,t) 
            else:
                if self.STM_logLikely[1] > self.transfer_threshold:
                    self.LTM = np.vstack([self.LTM,self.STM[i,:]])
                    if len(self.STM.shape) > 1:
                        STM_purge_ids.append(self.STM[i,0])
                    else:
                        STM_purge_ids.append(self.STM[0])
                    self.new_config()
                    self.update_LTM_logLikely(detections,pose,t) 

        if len(STM_purge_ids) > 0:
            self.STM = self.STM[np.where(self.STM[:,0] not in STM_purge_ids)]
            self.STM_logLikely = self.STM_logLikely[np.where(self.STM_logLikely[:,0] not in STM_purge_ids)]

        #finally, all voxels in the LTM with log odds value below a threshold are deleted from the world representation
        LTM_purge_ids = []
        #print("logLikely: ",self.LTM_logLikely)
        #print("LTM: ",self.LTM)
        for i in range(len(self.LTM)):
            if isinstance(self.LTM,list):
                self.LTM = np.array(self.LTM)
            if len(self.LTM.shape) > 1:
                log_likely_idx = [k for k,x in enumerate(self.LTM_logLikely[:,0]) if x == self.LTM[i,0]][0]
            else:
                if len(self.LTM_logLikely.shape) > 1:
                    log_likely_idx = [k for k,x in enumerate(self.LTM_logLikely[:,0]) if x == self.LTM[0]][0]
                else:
                    log_likely_idx = 0
            if len(self.LTM_logLikely.shape) > 1:
                #print("self.LTM_logLikely.shape: ",self.LTM_logLikely.shape)
                tmp = self.LTM_logLikely[log_likely_idx,1]
                if isinstance(tmp,np.ndarray):
                    tmp = tmp[0]
                if  tmp < self.rejection_threshold:
                    LTM_purge_ids.append(self.LTM_logLikely[log_likely_idx,0])
            else:
                if self.LTM_logLikely[1] < self.rejection_threshold:
                    LTM_purge_ids.append(self.LTM_logLikely[0])

        if len(LTM_purge_ids) > 0:
            if isinstance(self.LTM,list):
                self.LTM = np.array(self.LTM)
            if len(self.LTM.shape) > 1:
                self.LTM = np.array([x for x in self.LTM if x[0] not in LTM_purge_ids])
                self.LTM_logLikely = np.array([x for x in self.LTM_logLikely if x[0] not in LTM_purge_ids])
            else:
                if LTM_purge_ids[0] == self.LTM[0]:
                    self.LTM = []
                    self.LTM_logLikely = []
                else:
                    raise OSError

        self.track_activeMaps(pose,pose_covar,t,detections)

        return [x for x in detections if x['clique_id'] in self.get_best_lms()]

