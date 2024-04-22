from numpy import log, exp, log1p
import numpy as np

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

class PersistenceFilter:
  def __init__(self, lambda_u, num_features=2, initialization_time=0.0):
    #A function that returns the natural logarithm of the survival function S_T()
    self._log_survival_function = lambda t: -lambda_u*t
        
    #The timestamp at which this feature was first observed
    self._initialization_time = [initialization_time for i in range(num_features)]
    self.init_tstep = initialization_time 

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
        
  def update(self, detector_output, observation_time, P_M, P_F):
    
    list_of_detected_features = []
    for i in range(len(self._log_likelihood)):
        if(detector_output[i] == 1):
            list_of_detected_features.append(i)
    
    # do clique filtering
    if self._log_clique_lower_evidence_sum is not None:
        self._log_clique_lower_evidence_sum = logsum(self._log_clique_lower_evidence_sum, np.sum([el for el in self._log_likelihood]) + 
                                                self._shifted_logdF(observation_time, max(self._last_observation_time), np.asarray(self._last_observation_time).argmax())) + np.sum([(log(P_F) if el else log(1 - P_F)) for el in detector_output])
    else:
        self._log_clique_lower_evidence_sum = np.sum([(log(P_F) if el else log(1 - P_F)) for el in detector_output]) + log1p(-exp(self._shifted_log_survival_function(observation_time, 0)))
    
    
    # do pairwise filtering
    #print("len(self_log_likelihood): {}".format(len(self._log_likelihood))) 
    #print("self._last_observation_time: ",self._last_observation_time)
    for i in range(len(self._log_likelihood)):
        #Update the lower sum LY
        if self._log_lower_evidence_sum[i,i] is not None:
            #_log_lower_evidence_sum has been previously initialized, so just update it in the usual way
            self._log_lower_evidence_sum[i,i] = logsum(self._log_lower_evidence_sum[i,i], self._log_likelihood[i] + self._shifted_logdF(observation_time, self._last_observation_time[i], i)) + \
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
                self._log_lower_evidence_sum[i,j] = logsum(self._log_lower_evidence_sum[i,j], self._log_likelihood[i] + self._log_likelihood[j] + self._shifted_logdF(observation_time, max(self._last_observation_time[i], self._last_observation_time[j]), i)) +\
                    (log(P_F) if detector_output[i] else log(1 - P_F)) + (log(P_F) if detector_output[j] else log(1 - P_F))
            else:
                #This is the first observation we've incorporated; initialize the logarithm of lower running sum here
                self._log_lower_evidence_sum[i,j] = (log(P_F) if detector_output[i] else log(1 - P_F)) + (log(P_F) if detector_output[j] else log(1 - P_F)) + log1p(-exp(self._shifted_log_survival_function(observation_time, i)))
            self._log_lower_evidence_sum[j,i] = self._log_lower_evidence_sum[i,j]
            
            #Post-condition: at this point, self._log_lower_evidence_sum is a properly-initialized value

                                          
        #Update the measurement likelihood pY_tN
        self._log_likelihood[i] = self._log_likelihood[i] + (log(1.0 - P_M) if detector_output[i] else log(P_M))
        
        #Update the last observation time
        if i in list_of_detected_features:
            self._last_observation_time[i] = observation_time

        #Compute the marginal (evidence) probability pY
        self._log_evidence[i,i] = logsum(self._log_lower_evidence_sum[i,i], self._log_likelihood[i] + self._shifted_log_survival_function(self._last_observation_time[i], i))
        for j in range(i):
            if(j == i):
                continue
            self._log_evidence[i,j] = logsum(self._log_lower_evidence_sum[i,j], self._log_likelihood[i] + self._log_likelihood[j] + self._shifted_log_survival_function(self._last_observation_time[i], i))
            self._log_evidence[j,i] = self._log_evidence[i,j]
            
    self._clique_likelihood = np.sum(self._log_likelihood)
    
    self._log_clique_evidence = logsum(self._log_clique_lower_evidence_sum, np.sum(self._log_likelihood) + self._shifted_log_survival_function(self._last_observation_time[np.asarray(self._last_observation_time).argmax()], np.asarray(self._last_observation_time).argmax()))

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
        if prediction_time < self.init_tstep:
          return 0
        else:
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
  