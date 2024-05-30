    def estimate_growth_state(self,t,id_,detections_t,posterior): 
        #print("estimating growth state of clique {}....".format(id_))
        #need to get time in the current growth state, then compare to T_nu_lmType  

        global_t = self.current_exp*self.sim_length + t 

        if id_ in self.cone_ids:
            T_nu = self.T_nu_cone  
        elif id_ in self.tree_ids: 
            T_nu = self.T_nu_tree  
            
        if global_t == 0:
            prev_gstates_id = []
            d_t = 0 
        else:
            prev_gstates_id = self.growth_state_estimates[id_][:global_t]
            if self.last_index_of_change(id_,t,prev_gstates_id) > global_t:
                print("self.last_index_of_change: {}, global_t: {}".format(self.last_index_of_change(id_,t,prev_gstates_id),global_t))
                print("this isnt possible")
                raise OSError 
            #print("self.current_exp*self.sim_length: {}, self.last_index_of_change: {}".format((self.current_exp)*self.sim_length, self.last_index_of_change(id_,t,prev_gstates_id)))
            d_t = global_t - self.last_index_of_change(id_,t,prev_gstates_id)
    
        #print("this is d_t: ",d_t)
        if d_t < 0 or d_t > global_t: 
            print("this is d_t: {}, and this is global_t: {}".format(d_t,global_t)) 
            raise OSError 
        
        if global_t > 0:
            current_gstate = self.growth_state_estimates[id_][global_t-1]
        else:
            current_gstate = 1 

        if id_ in self.tree_ids:
            if current_gstate == 0 and posterior > self.acceptance_threshold:
                #print("woops we thought this was dead... we probably missed up bc the posterior is really high") 
                return self.sample_gstate(id_) 
            elif current_gstate == 0 and posterior < self.acceptance_threshold: 
                #print("nah this is prolly dead fr")
                return 0 
            elif 0 < current_gstate:
                #print("in this statement! 0 < current_gstate")
                if current_gstate + 1 <= 3:
                    #print("current_gstate+1: ",current_gstate + 1)
                    next_gstate = current_gstate + 1
                else:
                    #print("next_gstate is one")
                    next_gstate = 1
            elif current_gstate < 0:
                raise OSError 
        elif id_ in self.cone_ids:
            if current_gstate == 0:
                next_gstate = 1 
            elif current_gstate < 0:
                raise OSError 
            else:
                next_gstate = 0
        else:
            print("this is neither a tree or a cone:",id_)
            print("self.cone_ids: ",self.cone_ids)
            print("self.tree_ids: ",self.tree_ids)
            raise OSError 
        
        #print("this is current gstate: ",current_gstate)
        #print("this is next gstate: ",next_gstate)

        if global_t < 100 and current_gstate > 1:
            print(self.growth_state_estimates[id_])
            raise OSError 
        
        current_descriptors = self.get_feature_descriptors(detections_t)

        if current_gstate not in self.cone_feature_description_cache.keys() and id_ in self.cone_ids: 
            self.cone_feature_description_cache[current_gstate] = []
        elif current_gstate not in self.tree_feature_description_cache.keys() and id_ in self.tree_ids:  
            self.tree_feature_description_cache[current_gstate] = []

        if id_ in self.cone_ids: 
            if isinstance(self.cone_feature_description_cache[current_gstate],list): 
                 self.cone_feature_description_cache[current_gstate] = np.array(self.cone_feature_description_cache[current_gstate])
            if self.cone_feature_description_cache[current_gstate].size > 0:
                feature_similarity = self.determine_similarity(self.cone_feature_description_cache[current_gstate],current_descriptors) 
            else:
                feature_similarity = 0 
        elif id_ in self.tree_ids:
            if isinstance(self.tree_feature_description_cache[current_gstate],list): 
                self.tree_feature_description_cache[current_gstate] = np.array(self.tree_feature_description_cache[current_gstate])
            if self.tree_feature_description_cache[current_gstate].size > 0:
                feature_similarity = self.determine_similarity(self.tree_feature_description_cache[current_gstate],current_descriptors) 
            else:
                feature_similarity = 0 
        else:
            raise OSError 
        
        #Note: LOWER feature_similarity value implies 
        #print("this is feature_similarity: ",feature_similarity)
        #print("this is d_t: {} and T_nu: {}".format(d_t,T_nu))

        if d_t <= T_nu: 
            if d_t < T_nu*.05:
                #this is rachet ngl
                return current_gstate
            #print("the time for this growth state has not yet been exceeded")
            if self.hamming_similarity_val < feature_similarity: 
                #print("self.hamming_similarity_val:",self.hamming_similarity_val)
                #print("these features are different ... theres probably been a state change") 
                return next_gstate 
            else: 
                if feature_similarity < self.growth_state_relevancy_level: 
                    #print("these are very similar, adding these descriptors!")
                    if id_ in self.cone_ids: 
                        if self.cone_feature_description_cache[current_gstate].size > 0:
                            self.cone_feature_description_cache[current_gstate] = np.vstack((self.cone_feature_description_cache[current_gstate],current_descriptors))
                        else:
                            self.cone_feature_description_cache[current_gstate] = current_descriptors 
                    elif id_ in self.tree_ids: 
                        if self.tree_feature_description_cache[current_gstate].size > 0: 
                            self.tree_feature_description_cache[current_gstate] = np.vstack((self.tree_feature_description_cache[current_gstate],current_descriptors))
                        else:
                            self.tree_feature_description_cache[current_gstate] = current_descriptors
                    else:
                        raise OSError 
                    return current_gstate 
                elif self.growth_state_relevancy_level <= feature_similarity: 
                    #print("the features are different ... sample the gstate using the current time") 
                    if id_ in self.cone_ids: 
                        sampled_gstate = self.sample_gstate(id_,x=d_t/self.sim_length)
                    else:
                        sampled_gstate = self.sample_gstate(id_)
                    '''
                    if sampled_gstate == 0:
                        print("WARNING! sampling function thinks this clique should not persist!") 
                    '''
                    return sampled_gstate 
        else:       
            #print("the time in this growth state has been exceeded!") 
            if self.growth_state_relevancy_level < feature_similarity:
                #however the features are quite different 
                if id_ in self.cone_ids: 
                    self.exceeded_Tnu_cone_tsteps += 1 
                    if 10 < self.exceeded_Tnu_cone_tsteps:
                        self.T_nu_cone -= self.T_nu_cone*0.05
                        self.exceeded_Tnu_cone_tsteps = 0
                else: 
                    self.exceeded_Tnu_tree_tsteps += 1 
                    if 10 < self.exceeded_Tnu_tree_tsteps: 
                        self.T_nu_tree -= self.T_nu_tree*0.05 
                        self.exceeded_Tnu_tree_tsteps = 0 
                return next_gstate
            else: 
                #print("the features are quite similar ...") 
                #increase the mean of the distribution for this clique state 
                if id_ in self.cone_ids: 
                    self.T_nu_cone += self.T_nu_cone*0.05
                else: 
                    self.T_nu_tree += self.T_nu_tree*0.05
                return current_gstate