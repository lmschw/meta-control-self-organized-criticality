import random
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


from enums.EnumDistributionType import DistributionType
from enums.EnumEventEffect import EventEffect
from enums.EnumEventSelectionType import EventSelectionType
from enums.EnumColourType import ColourType

import events.BaseEvent as BaseEvent

import services.ServiceOrientations as ServiceOrientations
import services.ServicePreparation as ServicePreparation
import services.ServiceVicsekHelper as ServiceVicsekHelper

class FoodEvent:
    """
    Representation of an event occurring at a specified time and place within the domain and affecting 
    a specified percentage of particles. After creation, the check()-method takes care of everything.
    """
    def __init__(self, startTimestep, amount, domainSize, areas=None, radius=None, stopMovement=True):
        """
        Creates an external stimulus event that affects part of the swarm at a given timestep.

        Params:
            - startTimestep (int): the first timestep at which the stimulus is presented and affects the swarm
            - duration (int): the number of timesteps during which the stimulus is present and affects the swarm
            - domainSize (tuple of floats): the size of the domain
            - eventEffect (EnumEventEffect): how the orientations should be affected
            - distributionType (DistributionType): whether the event is global or local in nature
            - areas (list of arrays containing [x_center, y_center, radius]) [optional]: where the event is supposed to take effect. Required for Local events
            - radius (float) [optional]: the event radius
            - noisePercentage (float, range: 0-100) [optional]: how much noise is added to the orientation determined by the event (only works for certain events)
            - blockValues (boolean) [optional]: whether the values (nsm, k, speed etc.) should be blocked after the update. By default False
            
        Returns:
            No return.
        """
        self.startTimestep = startTimestep
        self.maxAmount = amount
        self.amount = amount
        self.domainSize = domainSize
        self.areas = areas
        self.radius = radius
        self.stopMovement = stopMovement

        self.radius = self.areas[0][2]

        if radius:
            self.radius = radius

        self.duration = -1
        
    def getShortPrintVersion(self):
        return f"t{self.startTimestep}d{self.duration}e{self.eventEffect.val}a{self.angle}dt{self.distributionType.value}a{self.areas}"

    def getParameterSummary(self):
        summary = {"startTimestep": self.startTimestep,
            "amount": self.amount,
            "domainSize": self.domainSize.tolist(),
            "areas": self.areas,
            "radius": self.radius,
            "stopMovement": self.stopMovement,
            "duration": self.duration
            }

        return summary
    
    def check(self, totalNumberOfParticles, currentTimestep, positions, speeds, hungerLevels):
        """
        Checks if the event is triggered at the current timestep and executes it if relevant.

        Params:
            - totalNumberOfParticles (int): the total number of particles within the domain. Used to compute the number of affected particles
            - currentTimestep (int): the timestep within the experiment run to see if the event should be triggered
            - positions (array of tuples (x,y)): the position of every particle in the domain at the current timestep
            - orientations (array of tuples (u,v)): the orientation of every particle in the domain at the current timestep
            - nsms (array of NeighbourSelectionMechanisms): the neighbour selection mechanism currently selected by each individual at the current timestep
            - ks (array of int): the number of neighbours currently selected by each individual at the current timestep
            - speeds (array of float): the speed of every particle at the current timestep
            - dt (float) [optional]: the difference between the timesteps
            - activationTimeDelays (array of int) [optional]: the time delay for the updates of each individual
            - isActivationTimeDelayRelevantForEvent (boolean) [optional]: whether the event can affect particles that may not be ready to update due to a time delay. They may still be selected but will retain their current values
            - colourType (ColourType) [optional]: if and how particles should be encoded for colour for future video rendering

        Returns:
            The orientations of all particles - altered if the event has taken place, unaltered otherwise.
        """
        affected = np.full(totalNumberOfParticles, False)
        if self.startTimestep >= currentTimestep and self.amount > 0:
            self.timestep = currentTimestep
            # if self.timestep % 100 == 0:
            #     print(f"t={currentTimestep}")
            # if currentTimestep == self.startTimestep or currentTimestep == (self.startTimestep + self.duration):
            #     print(f"executing event at timestep {currentTimestep}")
            speeds, hungerLevels, affected = self.executeEvent(totalNumberOfParticles=totalNumberOfParticles, positions=positions, speeds=speeds, hungerLevels=hungerLevels)
        elif self.duration == -1:
            self.duration = currentTimestep-self.startTimestep
        return speeds, hungerLevels, affected
    
    def executeEvent(self, totalNumberOfParticles, positions, speeds, hungerLevels):
        """
        Executes the event.

        Params:
            - totalNumberOfParticles (int): the total number of particles within the domain. Used to compute the number of affected particles
            - positions (array of tuples (x,y)): the position of every particle in the domain at the current timestep
            - orientations (array of tuples (u,v)): the orientation of every particle in the domain at the current timestep
            - nsms (array of NeighbourSelectionMechanisms): the neighbour selection mechanism currently selected by each individual at the current timestep
            - ks (array of int): the number of neighbours currently selected by each individual at the current timestep
            - speeds (array of float): the speed of every particle at the current timestep
            - dt (float) [optional]: the difference between the timesteps
            - colourType (ColourType) [optional]: if and how particles should be encoded for colour for future video rendering

        Returns:
            The orientations, neighbour selection mechanisms, ks, speeds, blockedness and colour of all particles after the event has been executed.
        """
        posWithCenter = np.zeros((totalNumberOfParticles+1, 2))
        posWithCenter[:-1] = positions
        posWithCenter[-1] = self.getOriginPoint()
        rij2 = ServiceVicsekHelper.getDifferences(posWithCenter, self.domainSize)
        relevantDistances = rij2[-1][:-1] # only the comps to the origin and without the origin point
        candidates = (relevantDistances <= self.radius**2)
        affected = self.selectAffected(candidates, relevantDistances)

        speeds[affected] = 0
        hungerLevels[affected] += 1

        return speeds, hungerLevels, affected
    

    def selectAffected(self, candidates, rij2):
        """
        Determines which particles are affected by the event.

        Params:
            - candidates (array of boolean): which particles are within range, i.e. within the event radius
            - rij2 (array of floats): the distance squared of every particle to the event focus point

        Returns:
            Array of booleans representing which particles are affected by the event.
        """
        if self.amount < len(candidates):
            numberOfAffected = self.amount
        else:
            numberOfAffected = len(candidates)

        self.amount -=numberOfAffected

        preselection = candidates # default case, we take all the candidates

        indices = np.argsort(rij2)[:numberOfAffected]
        preselection = np.full(len(candidates), False)
        preselection[indices] = True

        return candidates & preselection
    
    def getOriginPoint(self):
        """
        Determines the point of origin of the event.

        Params:
            None

        Returns:
            The point of origin of the event in [X,Y]-coordinates.
        """

        return self.areas[0][:2]
