import modulation 
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np 

class MaxiumLikelihood(BaseEstimator, ClassifierMixin):  
    """AMCML classifies the modulation type of the input signal
    using the maxium likelihood classifier."""

    def __init__(self, snr, modulationPool):  
        
        self.snr = snr
        
        self.modulationPool = modulationPool
        
        self.sigma = np.sqrt(10**(-self.snr/10))/np.sqrt(2)
        
        self.nModulation = len(modulationPool)
        
        self.likelihood = np.zeros(self.nModulation)
        
        self.symbolList = []
        
        self.C0 = self.sigma**2
        self.C1 = 1.0 / (2*np.pi*self.sigma**2)
        
    def fit(self, X, y=None):
        
        for iModulation, modulationType in enumerate(self.modulationPool):
            symbol = modulation.getsymbol(modulationType)
            self.symbolList.append(symbol)
        
        return 
    
    def predict(self, X):
        
        result = np.zeros(X.shape[0], dtype=int)

        for k, sigIn in enumerate(X): 
            
            for iModulation, modulationType in enumerate(self.modulationPool):

                symbol = self.symbolList[iModulation]

                self.likelihood[iModulation] = self.computeLikelihood(sigIn, symbol, self.sigma)
            decision = np.argmax(self.likelihood)
            result[k] = decision

        return result

    def computeLikelihood(self, signal, alphabet, sigma):
        """Likelihood Function Calculate the log likelihood of signal belonging 
        to a set of bivariate normal distributions with specified alphabet(means) and sigma(standard deviation)."""
        
        dist = np.abs(signal[:, None] - alphabet[None, :]) ** 2
        
        dist = np.exp(-dist / 2 / self.C0) * self.C1
    
        likelihood = np.mean(dist, axis=1)

        likelihood = np.sum(np.log(likelihood + 0.01))

        return likelihood
