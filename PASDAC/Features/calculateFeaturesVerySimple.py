import numpy as np

def calculateFeaturesVerySimple(data = np.array([])):
    
    """Calculates features Mean, Variance

    Parameters
    ----------
        data:                   numpy array

    Return
    ------
        dict:
        - dict.mean:            double
        - dict.variance:        double

    """
    
    if data.size == 0:
        return 
    
    f = []
    rv = {}
    rv['mean'] = np.mean(data)
    rv['variance'] = np.var(data)
    rv['standard_deviation'] = np.std(data)
    f.append(rv['mean'])
    f.append(rv['variance'])
    f.append(rv['standard_deviation'])
    return f
    

if __name__ == "__main__":

    mean = 0
    std = 1 
    num_samples = 500
    samples = np.random.normal(mean, std, size=num_samples)
    data = np.array(samples)
    features = calculateFeaturesVerySimple(data)
    print (features)