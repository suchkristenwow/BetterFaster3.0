import numpy as np   
import math

def wrap2pi(alpha):
	return (alpha + np.pi) % 2.0 * np.pi - np.pi

def norm_pdf_multivariate(x, mu, sigma):
	#print("Drawing from norm pdf multivariate...")
	sigma = np.matrix(sigma)
	size = len(x)
	if size == len(mu) and (size, size) == sigma.shape:
		det = np.linalg.det(sigma)
		if det == 0:
			raise NameError("The covariance matrix can't be singular")
		norm_const = 1.0/ (math.pow((2*math.pi),float(size)/2) * np.sqrt(det))
		#print(norm_const)
		x_mu = np.matrix(x - mu)
		inv = sigma.I        
		#print("x_mu: {}, inv: {}, x_mu.T: {}".format(x_mu,inv,x_mu.T))
		try:
			result = math.pow(math.e, -0.5 * (x_mu @ inv @ x_mu.T))
			return norm_const * result
		except:
			return 0

	else:
		raise NameError("The dimensions of the input don't match")