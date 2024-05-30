from numpy import log, exp, log1p
import numpy as np

def logdiff(in1,in2):
    out = in1 + np.log1p(-np.exp(in2-in1))
    return out

def logsum(in1,in2):
    if in1 < in2:
        temp = in1
        in1 = in2
        in2 = temp
    out = in1 + np.log1p(-np.exp(in2 - in1))
    if np.isnan(out):
        raise OSError
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
    #self._log_evidence = np.array([[0.0 for i in range(num_features)] for j in range(num_features)])
    self._log_clique_evidence = 0.0
        
    # TESTING MEMORILESS PRIOR 
    self._shifted_log_survival_function = lambda t, i: self._log_survival_function(t - self._last_observation_time[i])
    
    #A function that computes the logarithm of the prior probability assigned to the range [t0, t1) by the shifted survival time prior p_T()
    self._shifted_logdF = lambda t1, t0, i: logdiff(self._shifted_log_survival_function(t0, i), self._shifted_log_survival_function(t1, i)) if t1 - t0 != 0 else 0.0
        
  def update(self, detector_output, observation_time, P_M, P_F):
    detected_indices = np.nonzero(detector_output)[0]  # Indices of detected features

    # Precompute log(P_F) and log(1 - P_F)
    log_P_F = np.log(P_F)
    log_1_minus_P_F = np.log(1 - P_F)

    # Update clique filtering
    if self._log_clique_lower_evidence_sum is not None:
        self._log_clique_lower_evidence_sum = np.logaddexp(
            self._log_clique_lower_evidence_sum,
            np.sum(self._log_likelihood) +
            self._shifted_logdF(observation_time, np.max(self._last_observation_time), np.argmax(self._last_observation_time))) + \
                np.sum([log_P_F if el else log_1_minus_P_F for el in detector_output])
    else:
        self._log_clique_lower_evidence_sum = np.sum([log_P_F if el else log_1_minus_P_F for el in detector_output]) + \
            np.log1p(-np.exp(self._shifted_log_survival_function(observation_time, 0)))
        if np.isnan(self._log_clique_lower_evidence_sum):
            raise OSError
        
    # Update pairwise filtering
    for i in range(len(self._log_likelihood)):
        if self._log_lower_evidence_sum[i,i] is not None:
            self._log_lower_evidence_sum[i,i] = np.logaddexp(
                self._log_lower_evidence_sum[i,i],
                self._log_likelihood[i] +
                self._shifted_logdF(observation_time, self._last_observation_time[i], i)) + \
                (log_P_F if detector_output[i] else log_1_minus_P_F)
        else:
            self._log_lower_evidence_sum[i,i] = (log_P_F if detector_output[i] else log_1_minus_P_F) + \
                np.log1p(-np.exp(self._shifted_log_survival_function(observation_time, i)))
            if np.isnan(self._log_lower_evidence_sum[i,i]):
                raise OSError

        for j in range(i):
            if (j==i):
                continue 
            if self._log_lower_evidence_sum[i,j] is not None:
                self._log_lower_evidence_sum[i,j] = np.logaddexp(
                    self._log_lower_evidence_sum[i,j],
                    self._log_likelihood[i] + self._log_likelihood[j] +
                    self._shifted_logdF(observation_time, np.max([self._last_observation_time[i], self._last_observation_time[j]]), i) +
                    (log_P_F if detector_output[i] else log_1_minus_P_F) +
                    (log_P_F if detector_output[j] else log_1_minus_P_F))
            else:
                self._log_lower_evidence_sum[i,j] = (log_P_F if detector_output[i] else log_1_minus_P_F) + \
                    (log_P_F if detector_output[j] else log_1_minus_P_F) + \
                    np.log1p(-np.exp(self._shifted_log_survival_function(observation_time, i)))
                if np.isnan(self._log_lower_evidence_sum[i,j]):
                    raise OSError
                self._log_lower_evidence_sum[j,i] = self._log_lower_evidence_sum[i,j]

        self._log_likelihood[i] += (np.log(1.0 - P_M) if detector_output[i] else np.log(P_M))

        if i in detected_indices:
            self._last_observation_time[i] = observation_time

    self._clique_likelihood = np.sum(self._log_likelihood)

    log_survival_term = self._shifted_log_survival_function(np.max(self._last_observation_time), np.argmax(self._last_observation_time))
    self._log_clique_evidence = np.logaddexp(self._log_clique_lower_evidence_sum, np.sum(self._log_likelihood) + log_survival_term)

    if np.isnan(self._log_clique_evidence):
        raise OSError

  def predict_clique_likelihood(self, prediction_time):
        #print("self._clique_likelihood: ",self._clique_likelihood)
        #print("self._log_clique_evidence: ",self._log_clique_evidence)
        survival_function_term = self._shifted_log_survival_function(prediction_time, np.asarray(self._last_observation_time).argmax())
        #print("survival_function_term: ",survival_function_term)
        return np.exp(self._clique_likelihood - self._log_clique_evidence + survival_function_term)  

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