"""Python implementation of the persistence filter algorithm."""
from numpy import log, exp, log1p
import numpy as np
import scipy.integrate as integrate
import sys
from multiprocessing import Pool

global verbosity 

def logdiff(in1,in2):
    out = in1 + np.log1p(-np.exp(in2-in1))
    return out

def logsum(in1,in2):
    if in1 < in2:
        temp = in1;
        in1 = in2;
        in2 = temp;
    out = in1 + np.log1p(-np.exp(in2 - in1))
    return out

def logS_T(lambda_u,t):
    #print("lambda_u: ",lambda_u)
    rv = -lambda_u * t 
    return rv 

class PersistenceFilterNd:
  def __init__(self, lambda_u, num_features=2, initialization_time=0.0):
    #A function that returns the natural logarithm of the survival function S_T()
    self.lambda_u = lambda_u
    if not isinstance(lambda_u,float):
        raise OSError 
    #The timestamp at which this feature was first observed
    self._initialization_time = [initialization_time for i in range(num_features)]
    
    #The timestamp of the last detector output for this feature
    self._last_observation_time = [initialization_time for i in range(num_features)]
        
    #The natural logarithm of the likelihood probability p(Y_{1:N} | t_N)
    self._log_likelihood = [0.0 for i in range(num_features)]
    self._clique_likelihood = np.sum(self._log_likelihood)

    #print("this is num features: ",num_features)
    #The natural logarithm of the lower partial sum L(Y_{1:N}).  Note that we initialize this value as 'None', since L(Y_{1:0}) = 0 (i.e. this value is zero at initialization), for which the logarithm is undefined.  We initialize this running sum after the incorporation of the first observation.
    self._log_lower_evidence_sum = np.array([[None for i in range(num_features)] for j in range(num_features)])
    
    self._log_clique_lower_evidence_sum = None
        
    #The natural logarithm of the marginal (evidence) probability p(Y_{1:N})
    self._log_evidence = np.array([[0.0 for i in range(num_features)] for j in range(num_features)])
    self._log_clique_evidence = 0.0
        
    #A function returning the value of the survival time prior based upon the ELAPSED time since the feature's instantiation
    #self._shifted_log_survival_function = lambda t, i: self._log_survival_function(t - self._initialization_time[i])
    # TESTING MEMORILESS PRIOR 
    #self._shifted_log_survival_function = lambda t, i: logS_T(t - self._last_observation_time[i])
    
    #A function that computes the logarithm of the prior probability assigned to the range [t0, t1) by the shifted survival time prior p_T()
    #self._shifted_logdF = lambda t1, t0, i: logdiff(self._shifted_log_survival_function(t0, i), self._shifted_log_survival_function(t1, i)) if t1 - t0 != 0 else 0.0
        
  def _shifted_logdF(self,t1,t0,i): 
    return logdiff(self._shifted_log_survival_function(t0, i), self._shifted_log_survival_function(t1, i)) if t1 - t0 != 0 else 0.0

  def _shifted_log_survival_function(self,t,i):
    #print("self.lambda_u: ",self.lambda_u)
    #print("shifted t: ",t-self._last_observation_time[i])
    return logS_T(self.lambda_u,t - self._last_observation_time[i])

  def pairwise_filtering(self,args):
        args[0] = i; args[1] = detector_output
        if i in range(len(detector_output)):
            #print("finding the log likelihood... this is i: ",i)
            #Update the lower sum LY
            if self._log_lower_evidence_sum[i,i] is not None:
                #_log_lower_evidence_sum has been previously initialized, so just update it in the usual way
                self._log_lower_evidence_sum[i,i] =logsum(self._log_lower_evidence_sum[i,i], self._log_likelihood[i] + self._shifted_logdF(observation_time, self._last_observation_time[i], i)) + \
                                 (log(P_F) if detector_output[i] else log(1 - P_F))
            else:
                #This is the first observation we've incorporated; initialize the logarithm of lower running sum here
                self._log_lower_evidence_sum[i,i] = (log(P_F) if detector_output[i] else log(1 - P_F)) + log1p(-exp(self._shifted_log_survival_function(observation_time, i)))
            # update joint distribution
            #print("lower evidence sum: {}".format(self._log_lower_evidence_sum))

            for j in range(i):
                if(j == i):
                    continue
                if self._log_lower_evidence_sum[i,j] is not None:
                #_log_lower_evidence_sum has been previously initialized, so just update it in the usual way
                    self._log_lower_evidence_sum[i,j] =logsum(self._log_lower_evidence_sum[i,j], self._log_likelihood[i] + self._log_likelihood[j] + self._shifted_logdF(observation_time, max(self._last_observation_time[i], self._last_observation_time[j]), i)) +\
                        (log(P_F) if detector_output[i] else log(1 - P_F)) + (log(P_F) if detector_output[j] else log(1 - P_F))
                else:
                    #This is the first observation we've incorporated; initialize the logarithm of lower running sum here
                    self._log_lower_evidence_sum[i,j] = (log(P_F) if detector_output[i] else log(1 - P_F)) + (log(P_F) if detector_output[j] else log(1 - P_F)) + log1p(-exp(self._shifted_log_survival_function(observation_time, i)))
                self._log_lower_evidence_sum[j,i] = self._log_lower_evidence_sum[i,j]
                
                #Post-condition: at this point, self._log_lower_evidence_sum is a properly-initialized value
                #print("this is the joint clique evidence sum: {}".format(self._log_lower_evidence_sum))

            #Update the measurement likelihood pY_tN
            self._log_likelihood[i] = self._log_likelihood[i] + (log(1.0 - P_M) if detector_output[i] else log(P_M))
            #print("log likelihood: {}".format(self._log_likelihood))

            #Update the last observation time
            if i in list_of_detected_features:
                self._last_observation_time[i] = observation_time

            #print("shifted_log_survival_function: ",self._shifted_log_survival_function(self._last_observation_time[i], i))
            #Compute the marginal (evidence) probability pY
            self._log_evidence[i,i] =logsum(self._log_lower_evidence_sum[i,i], self._log_likelihood[i] + self._shifted_log_survival_function(self._last_observation_time[i], i))
            for j in range(i):
                if(j == i):
                    continue
                self._log_evidence[i,j] =logsum(self._log_lower_evidence_sum[i,j], self._log_likelihood[i] + self._log_likelihood[j] + self._shifted_log_survival_function(self._last_observation_time[i], i))
                self._log_evidence[j,i] = self._log_evidence[i,j]

  def parallel_update(self, detector_output, observation_time, P_M, P_F):
    #print("updating....")
    list_of_detected_features = []
    for i in range(len(self._log_likelihood)):
        if i in range(len(detector_output)):
            if(detector_output[i] == 1):
                list_of_detected_features.append(i)
    #print("list of detected features: ",list_of_detected_features)

    # do clique filtering
    if self._log_clique_lower_evidence_sum is not None:
        self._log_clique_lower_evidence_sum =logsum(self._log_clique_lower_evidence_sum, np.sum([el for el in self._log_likelihood]) + 
                                                self._shifted_logdF(observation_time, max(self._last_observation_time), np.asarray(self._last_observation_time).argmax())) + np.sum([(log(P_F) if el else log(1 - P_F)) for el in detector_output])
    else:
        self._log_clique_lower_evidence_sum = np.sum([(log(P_F) if el else log(1 - P_F)) for el in detector_output]) + log1p(-exp(self._shifted_log_survival_function(observation_time, 0)))
    #print("this is lower evidence sum: {}".format(self._log_clique_lower_evidence_sum))
    
    pool = Pool(processes=len(self._log_likelihood))
    args = [(i,detector_output) for i in range(len(self._log_likelihood))]
    pool.starmap(self.pairwise_filtering,args)
    pool.close()
    
    self._clique_likelihood = np.sum(self._log_likelihood)
    #print("self._clique_likelihood: ",self._clique_likelihood)

    self._log_clique_evidence =logsum(self._log_clique_lower_evidence_sum, np.sum(self._log_likelihood) + self._shifted_log_survival_function(self._last_observation_time[np.asarray(self._last_observation_time).argmax()], np.asarray(self._last_observation_time).argmax()))

  def update(self, detector_output, observation_time, P_M, P_F):
    #print("updating....")
    list_of_detected_features = []
    for i in range(len(self._log_likelihood)):
        if i in range(len(detector_output)):
            if(detector_output[i] == 1):
                list_of_detected_features.append(i)

    # do clique filtering
    if self._log_clique_lower_evidence_sum is not None:
        self._log_clique_lower_evidence_sum =logsum(self._log_clique_lower_evidence_sum, np.sum([el for el in self._log_likelihood]) + 
                                                self._shifted_logdF(observation_time, max(self._last_observation_time), np.asarray(self._last_observation_time).argmax())) + np.sum([(log(P_F) if el else log(1 - P_F)) for el in detector_output])
    else:
        self._log_clique_lower_evidence_sum = np.sum([(log(P_F) if el else log(1 - P_F)) for el in detector_output]) + log1p(-exp(self._shifted_log_survival_function(observation_time, 0)))
    #print("this is lower evidence sum: {}".format(self._log_clique_lower_evidence_sum))
    
    # do pairwise filtering
    for i in range(len(self._log_likelihood)):
        if i in range(len(detector_output)):
            #print("finding the log likelihood... this is i: ",i)
            #Update the lower sum LY
            if self._log_lower_evidence_sum[i,i] is not None:
                #_log_lower_evidence_sum has been previously initialized, so just update it in the usual way
                self._log_lower_evidence_sum[i,i] =logsum(self._log_lower_evidence_sum[i,i], self._log_likelihood[i] + self._shifted_logdF(observation_time, self._last_observation_time[i], i)) + \
                                 (log(P_F) if detector_output[i] else log(1 - P_F))
            else:
                #This is the first observation we've incorporated; initialize the logarithm of lower running sum here
                self._log_lower_evidence_sum[i,i] = (log(P_F) if detector_output[i] else log(1 - P_F)) + log1p(-exp(self._shifted_log_survival_function(observation_time, i)))
            # update joint distribution
            #print("lower evidence sum: {}".format(self._log_lower_evidence_sum))

            for j in range(i):
                if(j == i):
                    continue
                if self._log_lower_evidence_sum[i,j] is not None:
                #_log_lower_evidence_sum has been previously initialized, so just update it in the usual way
                    self._log_lower_evidence_sum[i,j] =logsum(self._log_lower_evidence_sum[i,j], self._log_likelihood[i] + self._log_likelihood[j] + self._shifted_logdF(observation_time, max(self._last_observation_time[i], self._last_observation_time[j]), i)) +\
                        (log(P_F) if detector_output[i] else log(1 - P_F)) + (log(P_F) if detector_output[j] else log(1 - P_F))
                else:
                    #This is the first observation we've incorporated; initialize the logarithm of lower running sum here
                    self._log_lower_evidence_sum[i,j] = (log(P_F) if detector_output[i] else log(1 - P_F)) + (log(P_F) if detector_output[j] else log(1 - P_F)) + log1p(-exp(self._shifted_log_survival_function(observation_time, i)))
                self._log_lower_evidence_sum[j,i] = self._log_lower_evidence_sum[i,j]
                
                #Post-condition: at this point, self._log_lower_evidence_sum is a properly-initialized value
                #print("this is the joint clique evidence sum: {}".format(self._log_lower_evidence_sum))

            #Update the measurement likelihood pY_tN
            self._log_likelihood[i] = self._log_likelihood[i] + (log(1.0 - P_M) if detector_output[i] else log(P_M))
            #print("log likelihood: {}".format(self._log_likelihood))

            #Update the last observation time
            if i in list_of_detected_features:
                self._last_observation_time[i] = observation_time

            #print("shifted_log_survival_function: ",self._shifted_log_survival_function(self._last_observation_time[i], i))
            #Compute the marginal (evidence) probability pY
            self._log_evidence[i,i] =logsum(self._log_lower_evidence_sum[i,i], self._log_likelihood[i] + self._shifted_log_survival_function(self._last_observation_time[i], i))
            for j in range(i):
                if(j == i):
                    continue
                self._log_evidence[i,j] =logsum(self._log_lower_evidence_sum[i,j], self._log_likelihood[i] + self._log_likelihood[j] + self._shifted_log_survival_function(self._last_observation_time[i], i))
                self._log_evidence[j,i] = self._log_evidence[i,j]
            
    #print("log evidence: ",self._log_evidence)

    self._clique_likelihood = np.sum(self._log_likelihood)
    #print("self._clique_likelihood: ",self._clique_likelihood)

    self._log_clique_evidence =logsum(self._log_clique_lower_evidence_sum, np.sum(self._log_likelihood) + self._shifted_log_survival_function(self._last_observation_time[np.asarray(self._last_observation_time).argmax()], np.asarray(self._last_observation_time).argmax()))

  def predict(self, prediction_time, feature_id, feature_id2=None):
    """Compute the posterior persistence probability p(X_t = 1 | Y_{1:N}).

    Args:
      prediction_time:  A floating-point value in the range
        [last_observation_time, infty) indicating the time t
        for which to compute the posterior survival belief p(X_t = 1 | Y_{1:N})

    Returns:
      A floating-point value in the range [0, 1] giving the
      posterior persistence probability p(X_t = 1 | Y_{1:N}).
    """
    if feature_id2 == None:
        return exp(self._log_likelihood[feature_id] - self._log_evidence[feature_id,feature_id] + self._shifted_log_survival_function(prediction_time, feature_id))
    else:
        return exp(self._log_likelihood[feature_id] + self._log_likelihood[feature_id2] - self._log_evidence[feature_id,feature_id2] + self._shifted_log_survival_function(prediction_time, feature_id))
  
  def predict_clique_likelihood(self, prediction_time):
    '''
    print("predicting clique likelihood: ",self._clique_likelihood)
    print("predict_clique_likelihood: ",self._log_clique_evidence)
    print("shifted_log_survival_function: ",self._shifted_log_survival_function(prediction_time, np.asarray(self._last_observation_time).argmax()))
    '''
    return np.exp(self._clique_likelihood - self._log_clique_evidence + self._shifted_log_survival_function(prediction_time, np.asarray(self._last_observation_time).argmax()))

  @property
  def log_survival_function(self):
    return self._log

  @property
  def shifted_log_survival_function(self):
    return self._shifted_log_survival_function
    
  @property
  def last_observation_time(self):
    return self._last_observation_time

  @property
  def initialization_time(self):
    return self._initialization_time



class PersistenceFilterNdSensorDegredation:
  def __init__(self, log_survival_function, num_features=2, initialization_time=0.0):
    #A function that returns the natural logarithm of the survival function S_T()
    self._log_survival_function = log_survival_function
        
    #The timestamp at which this feature was first observed
    self._initialization_time = [initialization_time for i in range(num_features)]
    
    #The timestamp of the last detector output for this feature
    self._last_observation_time = [initialization_time for i in range(num_features)]
        
    #The natural logarithm of the likelihood probability p(Y_{1:N} | t_N)
    self._log_likelihood = [0.0 for i in range(num_features)]
    self._clique_likelihood = np.sum(self._log_likelihood)
        
    #The natural logarithm of the lower partial sum L(Y_{1:N}).  Note that we initialize this value as 'None', since L(Y_{1:0}) = 0 (i.e. this value is zero at initialization), for which the logarithm is undefined.  We initialize this running sum after the incorporation of the first observation.
    self._log_lower_evidence_sum = np.array([[None for i in range(num_features)] for j in range(num_features)])
    
    self._log_clique_lower_evidence_sum = None
        
    #The natural logarithm of the marginal (evidence) probability p(Y_{1:N})
    self._log_evidence = np.array([[0.0 for i in range(num_features)] for j in range(num_features)])
    self._log_clique_evidence = 0.0
        
    #A function returning the value of the survival time prior based upon the ELAPSED time since the feature's instantiation
    #self._shifted_log_survival_function = lambda t, i: self._log_survival_function(t - self._initialization_time[i])
    # TESTING MEMORILESS PRIOR 
    self._shifted_log_survival_function = lambda t, i: self._log_survival_function(t - self._last_observation_time[i])
    
    #A function that computes the logarithm of the prior probability assigned to the range [t0, t1) by the shifted survival time prior p_T()
    self._shifted_logdF = lambda t1, t0, i: logdiff(self._shifted_log_survival_function(t0, i), self._shifted_log_survival_function(t1, i)) if t1 - t0 != 0 else 0.0
        
  def update(self, detector_output, ranges_output, observation_time, P_M_function, P_F):
    
    list_of_detected_features = []
    for i in range(len(self._log_likelihood)):
        if(detector_output[i] == 1):
            list_of_detected_features.append(i)
        
    clique_detected = False
    if(len(list_of_detected_features)):
        clique_detected = True

    if(clique_detected):
        # branch update in favor of persistence
        # update likelihoods without integral
        sufficient_feature = np.random.choice(list_of_detected_features)
    else:
        sufficient_feature = np.random.randint(len(self._log_likelihood))

    # do clique filtering
    if self._log_clique_lower_evidence_sum is not None:
        self._log_clique_lower_evidence_sum =logsum(self._log_clique_lower_evidence_sum, np.sum([el for el in self._log_likelihood]) + 
                                                self._shifted_logdF(observation_time, max(self._last_observation_time), np.asarray(self._last_observation_time).argmax())) + np.sum([(log(P_F) if el else log(1 - P_F)) for el in detector_output])
    else:
        self._log_clique_lower_evidence_sum = np.sum([(log(P_F) if el else log(1 - P_F)) for el in detector_output]) + log1p(-exp(self._shifted_log_survival_function(observation_time, 0)))
    
    
    # do pairwise filtering
    for i in range(len(self._log_likelihood)):
        #Update the lower sum LY
        if self._log_lower_evidence_sum[i,i] is not None:
            #_log_lower_evidence_sum has been previously initialized, so just update it in the usual way
            self._log_lower_evidence_sum[i,i] =logsum(self._log_lower_evidence_sum[i,i], self._log_likelihood[i] + self._shifted_logdF(observation_time, self._last_observation_time[i], i)) + \
                             (log(P_F) if detector_output[i] else log(1 - P_F))
        else:
            #This is the first observation we've incorporated; initialize the logarithm of lower running sum here
            self._log_lower_evidence_sum[i,i] = (log(P_F) if detector_output[i] else log(1 - P_F)) + log1p(-exp(self._shifted_log_survival_function(observation_time, i)))
        # update joint distribution
        
        for j in range(i):
            if(j == i):
                continue
            if self._log_lower_evidence_sum[i,j] is not None:
            #_log_lower_evidence_sum has been previously initialized, so just update it in the usual way
                self._log_lower_evidence_sum[i,j] =logsum(self._log_lower_evidence_sum[i,j], self._log_likelihood[i] + self._log_likelihood[j] + self._shifted_logdF(observation_time, max(self._last_observation_time[i], self._last_observation_time[j]), i)) +\
                    (log(P_F) if detector_output[i] else log(1 - P_F)) + (log(P_F) if detector_output[j] else log(1 - P_F))
            else:
                #This is the first observation we've incorporated; initialize the logarithm of lower running sum here
                self._log_lower_evidence_sum[i,j] = (log(P_F) if detector_output[i] else log(1 - P_F)) + (log(P_F) if detector_output[j] else log(1 - P_F)) + log1p(-exp(self._shifted_log_survival_function(observation_time, i)))
            self._log_lower_evidence_sum[j,i] = self._log_lower_evidence_sum[i,j]
            
            #Post-condition: at this point, self._log_lower_evidence_sum is a properly-initialized value

                                          
        #Update the measurement likelihood pY_tN
        self._log_likelihood[i] = self._log_likelihood[i] + (log(1.0 - P_M_function(ranges_output[i])) if detector_output[i] else log(P_M_function(ranges_output[i])))
        
        #Update the last observation time
        if i in list_of_detected_features:
            self._last_observation_time[i] = observation_time

        #Compute the marginal (evidence) probability pY
        self._log_evidence[i,i] =logsum(self._log_lower_evidence_sum[i,i], self._log_likelihood[i] + self._shifted_log_survival_function(self._last_observation_time[i], i))
        for j in range(i):
            if(j == i):
                continue
            self._log_evidence[i,j] =logsum(self._log_lower_evidence_sum[i,j], self._log_likelihood[i] + self._log_likelihood[j] + self._shifted_log_survival_function(self._last_observation_time[i], i))
            self._log_evidence[j,i] = self._log_evidence[i,j]
            
    self._clique_likelihood = np.sum(self._log_likelihood)
    
    self._log_clique_evidence =logsum(self._log_clique_lower_evidence_sum, np.sum(self._log_likelihood) + self._shifted_log_survival_function(self._last_observation_time[np.asarray(self._last_observation_time).argmax()], np.asarray(self._last_observation_time).argmax()))

  def predict(self, prediction_time, feature_id, feature_id2=None):
    """Compute the posterior persistence probability p(X_t = 1 | Y_{1:N}).

    Args:
      prediction_time:  A floating-point value in the range
        [last_observation_time, infty) indicating the time t
        for which to compute the posterior survival belief p(X_t = 1 | Y_{1:N})

    Returns:
      A floating-point value in the range [0, 1] giving the
      posterior persistence probability p(X_t = 1 | Y_{1:N}).
    """
    if feature_id2 == None:
        return exp(self._log_likelihood[feature_id] - self._log_evidence[feature_id,feature_id] + self._shifted_log_survival_function(prediction_time, feature_id))
    else:
        return exp(self._log_likelihood[feature_id] + self._log_likelihood[feature_id2] - self._log_evidence[feature_id,feature_id2] + self._shifted_log_survival_function(prediction_time, feature_id))
  def predict_clique_likelihood(self, prediction_time):
    return np.exp(self._clique_likelihood - self._log_clique_evidence + self._shifted_log_survival_function(prediction_time, np.asarray(self._last_observation_time).argmax()))

  @property
  def log_survival_function(self):
    return self._log

  @property
  def shifted_log_survival_function(self):
    return self._shifted_log_survival_function
    
  @property
  def last_observation_time(self):
    return self._last_observation_time

  @property
  def initialization_time(self):
    return self._initialization_time





class FernandoPersistanceFilter:
    def __init__(self, log_survival_function, num_features=2, initialization_time=0.0, P_False_detection=0.01, P_Miss_detection=0.3):
        self._seperated_evidence = np.zeros(num_features)
        self._log_survival_function = log_survival_function
        self.corrolation_matrix = np.ones((num_features, num_features)) / (num_features - 1)
        
        # i conditioned on k
        self._conditional_prior = lambda t, i, k:logsum(self.corrolation_matrix[i, k] + np.log(1.0 - integrate.quad(lambda x: np.exp(self._log_survival_function(x)), 0, t)[0]), self.corrolation_matrix[k,i])
        
        self._joint_prior = lambda t, k: np.sum([ obj._conditional_prior(t, i, k) for i in range(num_features)])
        
        
        self._feature_evidence = np.zeros((num_features))
        self._feature_partial_evidence = np.zeros((num_features))
        self._feature_likelihoods = np.zeros((num_features))
        self._feature_marginal = np.zeros((num_features))
        
        self.sufficient_feature = 0
        self.num_features = num_features
        self.last_observation_time = initialization_time
        self.P_False_detection = P_False_detection
        self.P_Miss_detection = P_Miss_detection
        
    def update(self, detector_output, observation_time):
        list_of_detected_features = []
        for i in range(self.num_features):
            if(detector_output[i] == 1):
                list_of_detected_features.append(i)

        clique_detected = False
        if(len(list_of_detected_features)):
            clique_detected = True
        if(clique_detected):
            # branch update in favor of persistence
            # update likelihoods without integral
            self.last_observed_feature = np.random.choice(list_of_detected_features)
        else:
            self.last_observed_feature = np.random.randint(self.num_features)
        
        # update partial evidence
        for i in range(self.num_features):
            self._feature_partial_evidence[i] =logsum(logsum( self._feature_partial_evidence[i], self._feature_likelihoods[i] + np.log(integrate.quad(lambda x: np.exp(self._log_survival_function(x)), self.last_observation_time, observation_time)[0])), 2*np.log(self.P_False_detection))
            self._feature_evidence[i] = self._feature_partial_evidence[i] + 2 * np.log(self.P_Miss_detection)
            

        # update marginal likelihood
        for i in range(self.num_features):
            if np.isnan(self._feature_marginal[i]):
                self._feature_marginal[i] = 0
            added_portion = self._feature_likelihoods[i] + integrate.quad(lambda t: self._conditional_prior(t, i, self.last_observed_feature), self.last_observation_time, observation_time)[0]
            print(added_portion, self._feature_likelihoods[i], self._conditional_prior(observation_time, i, self.last_observed_feature))
            self._feature_marginal[i] =logsum(self._feature_marginal[i], added_portion)
        
        # update likelihood
        for i in range(self.num_features):
            self._feature_likelihoods[i] += np.log(1.0 - self.P_Miss_detection if detector_output[i] else self.P_Miss_detection)
        
        self.last_observation_time = observation_time
        
    def predict(self, time, i):
        return self._feature_likelihoods[i] + np.log(1.0 - integrate.quad(lambda t: np.exp(self._log_survival_function(t)), 0, time)[0]) + self._feature_marginal[self.last_observed_feature] - self._feature_evidence[i] - self._feature_evidence[self.last_observed_feature]
    
    

class CovisabilityPersistenceFilter:
  def __init__(self, log_survival_function, num_features=2, initialization_time=0.0):
    self.num_features = num_features
    #A function that returns the natural logarithm of the survival function S_T()
    self._log_survival_function = log_survival_function
        
    #The timestamp at which this feature was first observed
    self._initialization_time = [initialization_time for i in range(num_features)]
    
    #The timestamp of the last detector output for this feature
    self._last_observation_time = [initialization_time for i in range(num_features)]
    
    self.observation_intersection = np.ones((num_features, num_features))
    self.observation_union = np.ones((num_features, num_features)) * 2
    
    self.no_detect_intersection = np.ones((num_features, num_features))
    self.no_detect_union = np.ones((num_features, num_features)) * 2
        
    #The natural logarithm of the likelihood probability p(Y_{1:N} | t_N)
    self._log_likelihood = np.zeros((num_features, num_features))
    self._clique_likelihood = np.sum(self._log_likelihood)
        
    #The natural logarithm of the lower partial sum L(Y_{1:N}).  Note that we initialize this value as 'None', since L(Y_{1:0}) = 0 (i.e. this value is zero at initialization), for which the logarithm is undefined.  We initialize this running sum after the incorporation of the first observation.
    self._log_lower_evidence_sum = np.array([[None for i in range(num_features)] for j in range(num_features)])
    
    self._log_clique_lower_evidence_sum = None
        
    #The natural logarithm of the marginal (evidence) probability p(Y_{1:N})
    self._log_evidence = np.array([[0.0 for i in range(num_features)] for j in range(num_features)])
        
    #A function returning the value of the survival time prior based upon the ELAPSED time since the feature's instantiation
    #self._shifted_log_survival_function = lambda t, i: self._log_survival_function(t - self._initialization_time[i])
    # TESTING MEMORILESS PRIOR 
    self._shifted_log_survival_function = lambda t, i: self._log_survival_function(t - self._last_observation_time[i])
    
    #A function that computes the logarithm of the prior probability assigned to the range [t0, t1) by the shifted survival time prior p_T()
    self._shifted_logdF = lambda t1, t0, i: logdiff(self._shifted_log_survival_function(t0, i), self._shifted_log_survival_function(t1, i)) if t1 - t0 != 0 else 0.0
        
  def update(self, detector_output, observation_time, P_M, P_F):
    
    list_of_detected_features = []
    for i in range(len(self._log_likelihood)):
        if(detector_output[i] == 1):
            list_of_detected_features.append(i)
        
    clique_detected = False
    if(len(list_of_detected_features)):
        clique_detected = True
        
    temp_obs = np.zeros(( self.num_features, self.num_features))
    for i in range(self.num_features):
        if i in list_of_detected_features:
            temp_obs[i, :] += 1
            temp_obs[:, i] += 1
    self.observation_intersection[temp_obs > 1] += 1
    self.observation_union[temp_obs > 0] +=1
    self.no_detect_intersection[temp_obs < 1] += 1
    self.no_detect_union[temp_obs < 2] += 1
    
    temp_eye = np.eye(self.num_features)
    self.p_miss_matrix = self.observation_intersection / self.observation_union
    self.p_false_matrix = self.no_detect_intersection / self.no_detect_union
    self.p_miss_matrix[temp_eye == 1] = P_M
    self.p_false_matrix[temp_eye == 1] = P_F
    
    if(clique_detected):
        # branch update in favor of persistence
        # update likelihoods without integral
        self.sufficient_feature = np.random.choice(list_of_detected_features)
    else:
        self.sufficient_feature = np.random.randint(self.num_features)
    self._s = self.sufficient_feature
    
    
    
    # do clique filtering
    if self._log_clique_lower_evidence_sum is not None:
        self._log_clique_lower_evidence_sum =logsum(self._log_clique_lower_evidence_sum, np.sum([el for el in self._log_likelihood[:, self._s]]) + 
                                                self._shifted_logdF(observation_time, self._last_observation_time[self._s], self._s)) + np.sum([(log(1.0 - self.p_false_matrix[i, self._s]) if el else log(self.p_false_matrix[i, self._s])) for i, el in enumerate(detector_output)])
    else:
        self._log_clique_lower_evidence_sum = np.sum([(log(1.0 - self.p_false_matrix[i, self._s]) if el else log(self.p_false_matrix[i, self._s])) for i, el in enumerate(detector_output)]) + log1p(-exp(self._shifted_log_survival_function(observation_time, 0)))
    
    # START HERE
                                                     
                                                     
                                                     
    # do pairwise filtering
    for i in range(len(self._log_likelihood)):
        #Update the lower sum LY
        if self._log_lower_evidence_sum[i,i] is not None:
            #_log_lower_evidence_sum has been previously initialized, so just update it in the usual way
            self._log_lower_evidence_sum[i,i] =logsum(self._log_lower_evidence_sum[i,i], self._log_likelihood[i, i] + self._shifted_logdF(observation_time, self._last_observation_time[i], i)) + \
                             (log(P_F) if detector_output[i] else log(1 - P_F))
        else:
            #This is the first observation we've incorporated; initialize the logarithm of lower running sum here
            self._log_lower_evidence_sum[i,i] = (log(P_F) if detector_output[i] else log(1 - P_F)) + log1p(-exp(self._shifted_log_survival_function(observation_time, i)))
        # update joint distribution
                                                     
        
        for j in range(i):
            if(j == i):
                continue
            if self._log_lower_evidence_sum[i,j] is not None:
            #_log_lower_evidence_sum has been previously initialized, so just update it in the usual way
                self._log_lower_evidence_sum[i,j] =logsum(self._log_lower_evidence_sum[i,j], self._log_likelihood[i, j] + self._log_likelihood[j,j] + self._shifted_logdF(observation_time, max(self._last_observation_time[i], self._last_observation_time[j]), i)) +\
                    (log(1 - self.p_false_matrix[i, j]) if detector_output[i] else log(self.p_false_matrix[i, j])) + (log(P_F) if detector_output[j] else log(1 - P_F))
                self._log_lower_evidence_sum[j,i] =logsum(self._log_lower_evidence_sum[j,i], self._log_likelihood[j, i] + self._log_likelihood[i,i] + self._shifted_logdF(observation_time, max(self._last_observation_time[i], self._last_observation_time[j]), j)) +\
                    (log(1 - self.p_false_matrix[j, i]) if detector_output[j] else log(self.p_false_matrix[j, i])) + (log(P_F) if detector_output[i] else log(1 - P_F))
                
                
                                                     
                                                     
            else:
                #This is the first observation we've incorporated; initialize the logarithm of lower running sum here
                self._log_lower_evidence_sum[i,j] = (log(1.0 - self.p_false_matrix[i, j]) if detector_output[i] else log(self.p_false_matrix[i, j])) + (log(P_F) if detector_output[j] else log(1 - P_F)) + log1p(-exp(self._shifted_log_survival_function(observation_time, i)))
                self._log_lower_evidence_sum[j,i] = (log(1.0 - self.p_false_matrix[j, i]) if detector_output[j] else log(self.p_false_matrix[j, i])) + (log(P_F) if detector_output[i] else log(1 - P_F)) + log1p(-exp(self._shifted_log_survival_function(observation_time, j)))
                
            #self._log_lower_evidence_sum[j,i] = self._log_lower_evidence_sum[i,j]
            
            #Post-condition: at this point, self._log_lower_evidence_sum is a properly-initialized value

                                          
        #Update the measurement likelihood pY_tN
        for j in range(self.num_features):
            if i == j:
                self._log_likelihood[i, j] = self._log_likelihood[i, j] + (log(1.0 - P_M) if detector_output[i] else log(P_M))
            else:
                self._log_likelihood[i, j] = self._log_likelihood[i, j] + (log(self.p_miss_matrix[i,j]) if detector_output[i] else log(1.0 - self.p_miss_matrix[i,j]))
        
        
        #Update the last observation time
        if i in list_of_detected_features:
            self._last_observation_time[i] = observation_time

        #Compute the marginal (evidence) probability pY
        self._log_evidence[i,i] =logsum(self._log_lower_evidence_sum[i,i], self._log_likelihood[i, i] + self._shifted_log_survival_function(self._last_observation_time[i], i))
        for j in range(i):
            if(j == i):
                continue
            self._log_evidence[i,j] =logsum(self._log_lower_evidence_sum[i,j], self._log_likelihood[i, j] + self._log_likelihood[j,j] + self._shifted_log_survival_function(self._last_observation_time[i], i))
            self._log_evidence[j,i] =logsum(self._log_lower_evidence_sum[j,i], self._log_likelihood[j, i] + self._log_likelihood[i,i] + self._shifted_log_survival_function(self._last_observation_time[j], j))
            
    self._clique_likelihood = np.sum(self._log_likelihood[:, self._s])
    
    self._log_clique_evidence =logsum(self._log_clique_lower_evidence_sum, np.sum(self._log_likelihood[:, self._s]) + self._shifted_log_survival_function(self._last_observation_time[np.asarray(self._last_observation_time).argmax()], np.asarray(self._last_observation_time).argmax()))

  def predict(self, prediction_time, feature_id, feature_id2=None):
    """Compute the posterior persistence probability p(X_t = 1 | Y_{1:N}).

    Args:
      prediction_time:  A floating-point value in the range
        [last_observation_time, infty) indicating the time t
        for which to compute the posterior survival belief p(X_t = 1 | Y_{1:N})

    Returns:
      A floating-point value in the range [0, 1] giving the
      posterior persistence probability p(X_t = 1 | Y_{1:N}).
    """
    if feature_id2 == None:
        return exp(self._log_likelihood[feature_id, feature_id] - self._log_evidence[feature_id,feature_id] + self._shifted_log_survival_function(prediction_time, feature_id))
    else:
        return exp(self._log_likelihood[feature_id, feature_id2] + self._log_likelihood[feature_id2, feature_id2] - self._log_evidence[feature_id,feature_id2] + self._shifted_log_survival_function(prediction_time, feature_id2))
  def predict_clique_likelihood(self, prediction_time):
    return np.exp(self._clique_likelihood - self._log_clique_evidence + self._shifted_log_survival_function(prediction_time, self.sufficient_feature))

  @property
  def log_survival_function(self):
    return self._log

  @property
  def shifted_log_survival_function(self):
    return self._shifted_log_survival_function
    
  @property
  def last_observation_time(self):
    return self._last_observation_time

  @property
  def initialization_time(self):
    return self._initialization_time




class CovisabilityPersistenceFilter2:
  def __init__(self, log_survival_function, num_features=2, initialization_time=0.0):
    #A function that returns the natural logarithm of the survival function S_T()
    self._log_survival_function = log_survival_function
    self.num_features = num_features
        
    #The timestamp at which this feature was first observed
    self._initialization_time = [initialization_time for i in range(num_features)]
    
    #The timestamp of the last detector output for this feature
    self._last_observation_time = [initialization_time for i in range(num_features)]
        
        
    self.observation_intersection = np.ones((num_features, num_features))
    self.observation_union = np.ones((num_features, num_features)) * 2
    
    self.no_detect_intersection = np.ones((num_features, num_features))
    self.no_detect_union = np.ones((num_features, num_features)) * 2
        
    #The natural logarithm of the likelihood probability p(Y_{1:N} | t_N)
    self._log_likelihood = [0.0 for i in range(num_features)]
    self._clique_likelihood = np.sum(self._log_likelihood)
        
    #The natural logarithm of the lower partial sum L(Y_{1:N}).  Note that we initialize this value as 'None', since L(Y_{1:0}) = 0 (i.e. this value is zero at initialization), for which the logarithm is undefined.  We initialize this running sum after the incorporation of the first observation.
    self._log_lower_evidence_sum = np.array([[None for i in range(num_features)] for j in range(num_features)])
    
    self._log_clique_lower_evidence_sum = None
        
    #The natural logarithm of the marginal (evidence) probability p(Y_{1:N})
    self._log_evidence = np.array([[0.0 for i in range(num_features)] for j in range(num_features)])
        
    #A function returning the value of the survival time prior based upon the ELAPSED time since the feature's instantiation
    #self._shifted_log_survival_function = lambda t, i: self._log_survival_function(t - self._initialization_time[i])
    # TESTING MEMORILESS PRIOR 
    self._shifted_log_survival_function = lambda t, i: self._log_survival_function(t - self._last_observation_time[i])
    
    #A function that computes the logarithm of the prior probability assigned to the range [t0, t1) by the shifted survival time prior p_T()
    self._shifted_logdF = lambda t1, t0, i: logdiff(self._shifted_log_survival_function(t0, i), self._shifted_log_survival_function(t1, i)) if t1 - t0 != 0 else 0.0
    
    self.sufficient_feature = np.random.randint(len(self._log_likelihood))
    self._s = self.sufficient_feature
    self._log_clique_evidence = 0.0
        
  def update(self, detector_output, observation_time, P_M, P_F):
    
    list_of_detected_features = []
    for i in range(len(self._log_likelihood)):
        if(detector_output[i] == 1):
            list_of_detected_features.append(i)
        
    clique_detected = False
    if(len(list_of_detected_features)):
        clique_detected = True
        
        
    temp_obs = np.zeros(( self.num_features, self.num_features))
    for i in range(self.num_features):
        if i in list_of_detected_features:
            temp_obs[i, :] += 1
            temp_obs[:, i] += 1
    self.observation_intersection[temp_obs > 1] += 1
    self.observation_union[temp_obs > 0] +=1
    self.no_detect_intersection[temp_obs < 1] += 1
    self.no_detect_union[temp_obs < 2] += 1
    
    temp_eye = np.eye(self.num_features)
    self.p_miss_matrix = self.observation_intersection / self.observation_union
    self.p_false_matrix = self.no_detect_intersection / self.no_detect_union
    self.p_miss_matrix[temp_eye == 1] = P_M
    self.p_false_matrix[temp_eye == 1] = P_F
    
    if(clique_detected):
        # branch update in favor of persistence
        # update likelihoods without integral
        self.sufficient_feature = np.random.choice(list_of_detected_features)
    #else:
    #    self.sufficient_feature = np.random.randint(len(self._log_likelihood))
    self._s = self.sufficient_feature
    
    # do clique filtering
    if self._log_clique_lower_evidence_sum is not None:
        self._log_clique_lower_evidence_sum =logsum(self._log_clique_lower_evidence_sum, np.sum([el for el in self._log_likelihood]) + 
                                                self._shifted_logdF(observation_time, max(self._last_observation_time), np.asarray(self._last_observation_time).argmax())) + np.sum([(log(P_F) if el else log(1 - P_F)) for el in detector_output])
    else:
        self._log_clique_lower_evidence_sum = np.sum([(log(P_F) if el else log(1 - P_F)) for el in detector_output]) + log1p(-exp(self._shifted_log_survival_function(observation_time, 0)))
    
    
    # do pairwise filtering
    for i in range(len(self._log_likelihood)):
        #Update the lower sum LY
        if self._log_lower_evidence_sum[i,i] is not None:
            #_log_lower_evidence_sum has been previously initialized, so just update it in the usual way
            self._log_lower_evidence_sum[i,i] =logsum(self._log_lower_evidence_sum[i,i], self._log_likelihood[i] + self._shifted_logdF(observation_time, self._last_observation_time[i], i)) + \
                             (log(P_F) if detector_output[i] else log(1 - P_F))
        else:
            #This is the first observation we've incorporated; initialize the logarithm of lower running sum here
            self._log_lower_evidence_sum[i,i] = (log(P_F) if detector_output[i] else log(1 - P_F)) + log1p(-exp(self._shifted_log_survival_function(observation_time, i)))
        # update joint distribution
        
        for j in range(i):
            if(j == i):
                continue
            if self._log_lower_evidence_sum[i,j] is not None:
            #_log_lower_evidence_sum has been previously initialized, so just update it in the usual way
                self._log_lower_evidence_sum[i,j] =logsum(self._log_lower_evidence_sum[i,j], self._log_likelihood[i] + self._log_likelihood[j] + self._shifted_logdF(observation_time, max(self._last_observation_time[i], self._last_observation_time[j]), i)) +\
                    (log(P_F) if detector_output[i] else log(1 - P_F)) + (log(P_F) if detector_output[j] else log(1 - P_F))
            else:
                #This is the first observation we've incorporated; initialize the logarithm of lower running sum here
                self._log_lower_evidence_sum[i,j] = (log(P_F) if detector_output[i] else log(1 - P_F)) + (log(P_F) if detector_output[j] else log(1 - P_F)) + log1p(-exp(self._shifted_log_survival_function(observation_time, i)))
            self._log_lower_evidence_sum[j,i] = self._log_lower_evidence_sum[i,j]
            
            #Post-condition: at this point, self._log_lower_evidence_sum is a properly-initialized value

                                          
        #Update the measurement likelihood pY_tN
        
        if self._s in list_of_detected_features:
            self._log_likelihood[i] = self._log_likelihood[i] + (log(self.p_miss_matrix[i, self._s]) if detector_output[i] else log(1.0 - self.p_miss_matrix[i, self._s]))
        else:
            self._log_likelihood[i] = self._log_likelihood[i] + (log(1.0 - self.p_false_matrix[i, self._s]) if detector_output[i] else log(self.p_false_matrix[i, self._s]))
        
        #Update the last observation time
        if i in list_of_detected_features:
            self._last_observation_time[i] = observation_time

        #Compute the marginal (evidence) probability pY
        self._log_evidence[i,i] =logsum(self._log_lower_evidence_sum[i,i], self._log_likelihood[i] + self._shifted_log_survival_function(self._last_observation_time[i], i))
        for j in range(i):
            if(j == i):
                continue
            self._log_evidence[i,j] =logsum(self._log_lower_evidence_sum[i,j], self._log_likelihood[i] + self._log_likelihood[j] + self._shifted_log_survival_function(self._last_observation_time[i], i))
            self._log_evidence[j,i] = self._log_evidence[i,j]
            
    self._clique_likelihood = np.sum(self._log_likelihood)
    
    self._log_clique_evidence =logsum(self._log_clique_lower_evidence_sum, np.sum(self._log_likelihood) + self._shifted_log_survival_function(self._last_observation_time[np.asarray(self._last_observation_time).argmax()], np.asarray(self._last_observation_time).argmax()))

  def predict(self, prediction_time, feature_id, feature_id2=None):
    """Compute the posterior persistence probability p(X_t = 1 | Y_{1:N}).

    Args:
      prediction_time:  A floating-point value in the range
        [last_observation_time, infty) indicating the time t
        for which to compute the posterior survival belief p(X_t = 1 | Y_{1:N})

    Returns:
      A floating-point value in the range [0, 1] giving the
      posterior persistence probability p(X_t = 1 | Y_{1:N}).
    """
    if feature_id2 == None:
        return exp(self._log_likelihood[feature_id] - self._log_evidence[feature_id,feature_id] + self._shifted_log_survival_function(prediction_time, feature_id))
    else:
        return exp(self._log_likelihood[feature_id] + self._log_likelihood[feature_id2] - self._log_evidence[feature_id,feature_id2] + self._shifted_log_survival_function(prediction_time, feature_id))
  def predict_clique_likelihood(self, prediction_time):
    return np.exp(self._clique_likelihood - self._log_clique_evidence + self._shifted_log_survival_function(prediction_time, np.asarray(self._last_observation_time).argmax()))

  @property
  def log_survival_function(self):
    return self._log

  @property
  def shifted_log_survival_function(self):
    return self._shifted_log_survival_function
    
  @property
  def last_observation_time(self):
    return self._last_observation_time

  @property
  def initialization_time(self):
    return self._initialization_time
'''
  def logdiff(in1,in2):
    out = in1 + np.log1p(-np.exp(in2-in1))
    return out

  def logsum(in1,in2):
    out = in1 + np.log1p(-np.exp(in2 - in1))
    return out

  def log_general_purpose_survival_function(t,lambda_l,lambda_u):
    EI_lambda_l = tf.math.special.expint(lambda_l*t);
    EI_lambda_u = tf.math.special.expint(lambda_u*t);
    log_survival = logdiff(np.log(EI_lambda_l), np.log(EI_lambda_u))-np.log(np.log(lambda_u)-np.log(lambda_l));
    return log_survival 
'''

