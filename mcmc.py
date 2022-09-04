from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import gamma
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import numpy as np

#creates a discrete space for sampling
def discreteSpace(start, end):
    arr = []
    for i in range(start, end+1):
        arr.append(i)
    return arr

#this is the pi function referenced in literature
def targetFunction(x):
    return norm(x,1)

#uninformative prior
def priorDist(x):
    return uniform.pdf(x)


#the walk pulls random values from a standardized normal distribution by default
def randomWalk(dist = norm):
    rwlk = dist.rvs()
    return rwlk

def A(current, proposed, data):

    loglklhdProp = np.log(targetFunction(proposed).pdf(data)).sum()
    loglklhdCurr = np.log(targetFunction(current).pdf(data)).sum()

    lklhdProp = np.exp(loglklhdProp)
    lklhdCurr = np.exp(loglklhdCurr)
    
    #informative prior
    priorProp = priorDist(proposed)
    priorCurr = priorDist(current)

    probProp = lklhdProp*priorProp
    probCurr = lklhdCurr*priorCurr
    if probCurr == 0:
        if (max(targetFunction(proposed).pdf(data)) > max(targetFunction(current).pdf(data))):
            return proposed
        else:
            return current
        
    ratio = probProp / probCurr
    acceptanceRatio = min(ratio, 1)
    acceptanceThreshhold = uniform(0,1).rvs()

    if acceptanceThreshhold < acceptanceRatio:
        return proposed
    else:
        return current


def metropolis(sampleNumber):
    trueMean = 5
    sd = 3
    data = norm(trueMean,sd).rvs(30)
    data.sort()
    end = sampleNumber
    initVal = 7
    x = [0]*(end-1)
    x[0] = initVal
    for i in range(1,end-1):
        current = x[i-1]
        proposed = current + randomWalk()
        x[i] = A(current, proposed, data)
    
    
    linSpace = np.linspace(trueMean-5*sd, trueMean+5*sd, 30)
    burnin = 2000
    y = x[burnin:end]
    simulatedMean = np.mean(y)
    plt.plot(y, 'k.', label = "Mean = " + str(simulatedMean))
    plt.legend(loc="upper left")
    plt.savefig('mean dist', bbox_inches='tight')
    plt.show()
    plt.hist(y, bins = 20)
    plt.savefig('mean histo', bbox_inches='tight')
    plt.show()
    plt.plot(linSpace, norm(trueMean,sd).pdf(linSpace), label = "Data")
    plt.plot(linSpace, norm(simulatedMean,sd).pdf(linSpace), label = "Simulated Data")
    plt.legend(loc = "upper left")
    plt.savefig('data vs simulation', bbox_inches='tight')
    plt.show()



sampleNumber = 10000
metropolis(sampleNumber)
