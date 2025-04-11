import pandas as pd
import numpy as np

import services.ServiceOrientations as ServiceOrientations
import services.ServiceVicsekHelper as ServiceVicsekHelper
import services.ServiceMetric as ServiceMetric
import services.ServiceThresholdEvaluation as ServiceThresholdEvaluation
import services.ServiceSavedModel as ServiceSavedModel

import model.SwitchInformation as SwitchInformation

class VicsekWithNeighbourSelection():

    def __init__(self, domainSize, radius, noise, numberOfParticles,
                 speed=1, use_single_speed=True, degreesOfVision=2*np.pi, occlusion_active=False, 
                 returnHistories=True, logPath=None, logInterval=1):
        """
        Params:
            - domainSize (tuple of floats): the size of the domain
            - radius (float): the perception radius of the individuals
            - noise (float): the noise in the environment that is applied to the orientation of each particle at every timestep
            - numberOfParticles (int): how many particles are in the domain
            - speed (float) [optional]: the speed at which the particles move
            - use_single_speed (boolean) [optional]: whether all particles move at the same speed
            - degreesOfVision (float, range(0, 2pi)) [optional]: how much of their surroundings each individual is able to see. By default 2pi
            - activationTimeDelays (array of int) [optional]: how often each individual updates its orientation
            - isActivationTimeDelayRelevantForEvents (boolean) [optional]: whether an individual should also ignore events when it is not ready to update its orientation
            - colourType (ColourType) [optional]: if and how individuals should be coloured for future rendering
        """

        self.domainSize = np.asarray(domainSize)
        self.radius = radius
        self.noise = noise
        self.numberOfParticles = numberOfParticles
        self.speed = speed
        self.use_single_speed = use_single_speed
        self.degreesOfVision = degreesOfVision
        self.occlusion_active = occlusion_active

        self.returnHistories = returnHistories
        self.logPath = logPath
        self.logInterval = logInterval


    def getParameterSummary(self, asString=False):
        """
        Creates a summary of all the model parameters ready for use for conversion to JSON or strings.

        Parameters:
            - asString (bool, default False) [optional]: if the summary should be returned as a dictionary or as a single string
        
        Returns:
            A dictionary or a single string containing all model parameters.
        """
        summary = {"n": self.numberOfParticles,
                    "noise": self.noise,
                    "radius": self.radius,
                    "domainSize": self.domainSize.tolist(),
                    "tmax": self.tmax,
                    "dt": self.dt,
                    "degreesOfVision": self.degreesOfVision,
                    "speed": self.speed,
                    "use_single_speed": self.use_single_speed
                    }
        
        if asString:
            strPrep = [tup[0] + ": " + tup[1] for tup in summary.values()]
            return ", ".join(strPrep)
        return summary


    def initializeState(self):
        """
        Initialises the state of the swarm at the start of the simulation.

        Params:
            None
        
        Returns:
            Arrays of positions and orientations containing values for every individual within the system
        """
        positions = self.domainSize*np.random.rand(self.numberOfParticles,len(self.domainSize))
        orientations = ServiceOrientations.normalizeOrientations(np.random.rand(self.numberOfParticles, len(self.domainSize))-0.5)

        return positions, orientations
  
    def generateNoise(self):
        """
        Generates some noise based on the noise amplitude set at creation.

        Params:
            None

        Returns:
            An array with the noise to be added to each individual
        """
        return np.random.normal(scale=self.noise, size=(self.numberOfParticles, len(self.domainSize)))

    def calculateMeanOrientations(self, orientations, neighbours):
        """
        Computes the average of the orientations of all selected neighbours for every individual.

        Params:
            - orientations (array of floats): the orientation of every individual at the current timestep
            - neighbours (array of arrays of booleans): the identity of every neighbour of every individual

        Returns:
            An array of floats containing the new, normalised orientations of every individual
        """
        summedOrientations = np.sum(neighbours[:,:,np.newaxis]*orientations[np.newaxis,:,:],axis=1)
        return ServiceOrientations.normalizeOrientations(summedOrientations)

    def computeNewOrientations(self, neighbours, orientations):
        """
        Computes the new orientation of every individual based on the neighbour selection mechanisms, ks, time delays and Vicsek-like 
        averaging.
        Also sets the colours for ColourType.EXAMPLE.

        Params:
            - neighbours (array of arrays of booleans): the identity of every neighbour of every individual
            - positions (array of floats): the position of every individual at the current timestep
            - orientations (array of floats): the orientation of every individual at the current timestep
            - nsms (array of NeighbourSelectionMechanism): the neighbour selection mechanism used by every individual at the current timestep
            - ks (array of ints): the number of neighbours k used by every individual at the current timestep
            - activationTimeDelays (array of ints): at what rate updates are possible for every individual at the current timestep

        Returns:
            An array of floats representing the orientations of all individuals after the current timestep
        """

        oldOrientations = np.copy(orientations)

        orientations = self.calculateMeanOrientations(orientations, neighbours)
        orientations = ServiceOrientations.normalizeOrientations(orientations+self.generateNoise())
        
        return orientations
    
    def prepareSimulation(self, initialState, dt, tmax):
        """
        Prepares the simulation by initialising all necessary properties.

        Params:
            - initialState (tuple of arrays) [optional]: A tuple containing the initial positions of all particles, their initial orientations and their initial switch type values
            - dt (int) [optional]: time step
            - tmax (int) [optional]: the total number of time steps of the experiment

        Returns:
            Arrays containing the positions, orientations, neighbour selection mechanisms, ks, speeds and time delays.
        """
         # Preparations and setting of parameters if they are not passed to the method
        
        if any(ele is None for ele in initialState):
            positions, orientations = self.initializeState()
        else:
            positions, orientations = initialState

        # TODO implement speed generation
        speeds = np.full(self.numberOfParticles, self.speed)

        #print(f"t=pre, order={ServiceMetric.computeGlobalOrder(orientations)}")

        if dt is None and tmax is not None:
            dt = 1
        
        if tmax is None:
            tmax = (10**3)*dt
            dt = np.average(10**(-2)*(np.max(self.domainSize)/speeds))

        self.tmax = tmax
        self.dt = dt

        # Initialisations for the loop and the return variables
        self.numIntervals=int(tmax/dt+1)

        self.thresholdEvaluationChoiceValuesHistory = []  
        if self.returnHistories:
            self.positionsHistory = np.zeros((self.numIntervals,self.numberOfParticles,len(self.domainSize)))
            self.orientationsHistory = np.zeros((self.numIntervals,self.numberOfParticles,len(self.domainSize)))  

            self.positionsHistory[0,:,:]=positions
            self.orientationsHistory[0,:,:]=orientations

        return positions, orientations, speeds

    def updateHistoriesAndLogs(self, t, positions, orientations):
        if self.returnHistories:
            self.positionsHistory[t,:,:]=positions
            self.orientationsHistory[t,:,:]=orientations
        if self.logPath and t % self.logInterval == 0:
            switchTypeValues = None
            ServiceSavedModel.saveModelTimestep(timestep=t, 
                                                positions=positions, 
                                                orientations=orientations,
                                                path=self.logPath,
                                                switchValues=switchTypeValues,
                                                switchingActive=False)

    def simulate(self, initialState=(None, None, None), dt=None, tmax=None):
        """
        Runs the simulation experiment.
        First the parameters are computed if they are not passed. 
        Then the positions and orientations are computed for each particle at each time step.

        Params:
            - initialState (tuple of arrays) [optional]: A tuple containing the initial positions of all particles, their initial orientations and their initial switch type values
            - dt (int) [optional]: time step
            - tmax (int) [optional]: the total number of time steps of the experiment

        Returns:
            (times, positionsHistory, orientationsHistory), the history of the switchValues as a dictionary  and optionally coloursHistory. All except the switchValueHistory as ordered arrays so that they can be matched by index matching
        """
       
        positions, orientations, speeds = self.prepareSimulation(initialState=initialState, dt=dt, tmax=tmax)

        for t in range(self.numIntervals):
            self.t = t
            if t % 1000 == 0:
                print(f"t={t}/{self.tmax}")
            # if self.t % 100 == 0:
            #     print(f"{t}: {ServiceMetric.computeGlobalOrder(orientations)}")

            # all neighbours (including self)
            neighbours = ServiceVicsekHelper.getNeighboursWithLimitedVision(positions=positions, orientations=orientations, domainSize=self.domainSize,
                                                                            radius=self.radius, fov=self.degreesOfVision, occlusion_active=self.occlusion_active)

            orientations = self.computeNewOrientations(neighbours, orientations)

            positions += self.dt*(orientations.T * speeds).T
            positions += -self.domainSize*np.floor(positions/self.domainSize)

            self.updateHistoriesAndLogs(t=t,
                                        positions=positions,
                                        orientations=orientations)

            # if t % 500 == 0:
            #     print(f"t={t}, th={self.thresholdEvaluationMethod.name}, order={ServiceMetric.computeGlobalOrder(orientations)}")

        return (self.dt*np.arange(self.numIntervals), self.positionsHistory, self.orientationsHistory)
