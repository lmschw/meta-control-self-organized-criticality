import pandas as pd
import numpy as np
import random

from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumSwitchType import SwitchType
from enums.EnumColourType import ColourType
from enums.EnumThresholdEvaluationMethod import ThresholdEvaluationMethod

import services.ServiceOrientations as ServiceOrientations
import services.ServiceVicsekHelper as ServiceVicsekHelper
import services.ServiceMetric as ServiceMetric
import services.ServiceThresholdEvaluation as ServiceThresholdEvaluation

import model.SwitchInformation as SwitchInformation
from model.VicsekIndividualsMultiSwitchOscillation  import VicsekWithNeighbourSelectionOscillation

from events.FoodEvent import FoodEvent

class VicsekWithNeighbourSelectionOscillationFood(VicsekWithNeighbourSelectionOscillation):

    def __init__(self, domainSize, radius, noise, numberOfParticles, k, neighbourSelectionMechanism,
                 speed=1, switchSummary=None, events=None, degreesOfVision=2*np.pi, 
                 maxFood=50, foodAppearanceProbability=0.1, foodEvents=[], foodSourceAmount=10,
                 activationTimeDelays=[], isActivationTimeDelayRelevantForEvents=False, colourType=None, 
                 thresholdEvaluationMethod=ThresholdEvaluationMethod.LOCAL_ORDER, updateIfNoNeighbours=True,
                 individualistic_stress_delta=0.01, social_stress_delta=0.01, stress_num_neighbours=2):
        """
        Params:
            - domainSize (tuple of floats): the size of the domain
            - radius (float): the perception radius of the individuals
            - noise (float): the noise in the environment that is applied to the orientation of each particle at every timestep
            - numberOfParticles (int): how many particles are in the domain
            - k (int): how many neighbours should each individual consider (start value if k-switching is active)
            - neighbourSelectonMechansim (NeighbourSelectionMechanism): which neighbours each individual should consider (start value if nsm-switching is active)
            - speed (float) [optional]: the speed at which the particles move
            - switchSummary (SwitchSummary) [optional]: The switches that are available to the particles
            - events (list of BaseEvents or child classes) [optional]: the events that occur within the domain during the simulation
            - degreesOfVision (float, range(0, 2pi)) [optional]: how much of their surroundings each individual is able to see. By default 2pi
            - activationTimeDelays (array of int) [optional]: how often each individual updates its orientation
            - isActivationTimeDelayRelevantForEvents (boolean) [optional]: whether an individual should also ignore events when it is not ready to update its orientation
            - colourType (ColourType) [optional]: if and how individuals should be coloured for future rendering
        """
        super().__init__(domainSize=domainSize,
                         radius=radius,
                         noise=noise,
                         numberOfParticles=numberOfParticles,
                         k=k,
                         neighbourSelectionMechanism=neighbourSelectionMechanism,
                         speed=speed,
                         switchSummary=switchSummary,
                         events=events,
                         degreesOfVision=degreesOfVision,
                         individualistic_stress_delta=individualistic_stress_delta,
                         social_stress_delta=social_stress_delta,
                         stress_num_neighbours=stress_num_neighbours,
                         activationTimeDelays=activationTimeDelays,
                         isActivationTimeDelayRelevantForEvents=isActivationTimeDelayRelevantForEvents,
                         colourType=colourType,
                         thresholdEvaluationMethod=thresholdEvaluationMethod,
                         updateIfNoNeighbours=updateIfNoNeighbours)

        self.maxFood = maxFood
        self.foodAppearanceProbability = foodAppearanceProbability
        self.foodEvents = foodEvents
        self.foodSourceAmount = foodSourceAmount

    def getParameterSummary(self, asString=False):
        """
        Creates a summary of all the model parameters ready for use for conversion to JSON or strings.

        Parameters:
            - asString (bool, default False) [optional]: if the summary should be returned as a dictionary or as a single string
        
        Returns:
            A dictionary or a single string containing all model parameters.
        """
        summary = {"n": self.numberOfParticles,
                    "k": self.k,
                    "noise": self.noise,
                    "radius": self.radius,
                    "neighbourSelectionMechanism": self.neighbourSelectionMechanism.name,
                    "domainSize": self.domainSize.tolist(),
                    "tmax": self.tmax,
                    "dt": self.dt,
                    "degreesOfVision": self.degreesOfVision,
                    "activationTimeDelays": self.activationTimeDelays.tolist(),
                    "isActivationTimeDelayRelevantForEvents": self.isActivationTimeDelayRelevantForEvents,
                    "individualistic_stress_delta": self.individualistic_stress_delta,
                    "social_stress_delta": self.social_stress_delta,
                    "stress_num_neighbours": self.stress_num_neighbours
                    }

        if self.colourType != None:
            summary["colourType"] = self.colourType.value
            if self.exampleId != None:
                summary["exampleId"] = self.exampleId.tolist()

        if self.switchSummary != None:
            summary["switchSummary"] = self.switchSummary.getParameterSummary()

        if self.events:
            eventsSummary = []
            for event in self.events:
                eventsSummary.append(event.getParameterSummary())
            summary["events"] = eventsSummary

        if asString:
            strPrep = [tup[0] + ": " + tup[1] for tup in summary.values()]
            return ", ".join(strPrep)
        return summary

    def handleFoodEvents(self, t, positions, speeds, hungerLevels):
        """
        Handles all types of events.

        Params:
            - t (int): the current timestep
            - positions (array of (x,y)-coordinates): the position of every particle at the current timestep
            - orientations (array of (u,v)-coordinates): the orientation of every particle at the current timestep
            - nsms (array of NeighbourSelectionMechanism): how every particle selects its neighbours at the current timestep
            - ks (array of ints): how many neighbours each particle considers at the current timestep
            - speeds (array of floats): how fast each particle moves at the current timestep
            - activationTimeDelays (array of ints): how often a particle is ready to update its orientation

        Returns:
            Arrays containing the updates orientations, neighbour selecton mechanisms, ks, speeds, which particles are blocked from updating and the colours assigned to each particle.
        """

        overallAffected = []
        if self.foodEvents != None:
                for event in self.foodEvents:
                    speeds, hungerLevels, affected = event.check(self.numberOfParticles, t, positions, speeds, hungerLevels)
                    overallAffected.append(affected)
        # make sure that all affected particles have stopped and everyone else is moving
        speeds = np.full(self.numberOfParticles, self.speed)
        if len(self.foodEvents) > 0:
            flat = np.array(list(set(np.array(np.array(overallAffected).nonzero()[1]).flatten())))
            if len(flat) > 0:
                speeds[flat] = 0
        speeds = np.where(hungerLevels >= self.maxFood, self.speed, speeds)
        return speeds, hungerLevels
    
    def updateFoodEvents(self):
        if self.foodAppearanceProbability not in [None, 0]:
            rand = random.random()
            if rand < self.foodAppearanceProbability:
                foodEvent = FoodEvent(startTimestep=self.t,
                                      amount=self.foodSourceAmount,
                                      domainSize=self.domainSize,
                                      areas=[(random.random() * self.domainSize[0], random.random() * self.domainSize[1], self.radius)],
                                      radius=self.radius,
                                      stopMovement=True)
                self.foodEvents.append(foodEvent)


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
       
        positions, orientations, nsms, ks, speeds, activationTimeDelays, stressLevels = self.prepareSimulation(initialState=initialState, dt=dt, tmax=tmax)
        hungerLevels = np.full(self.numberOfParticles, self.maxFood)
        alive = np.full(self.numberOfParticles, True)
        self.hungerLevelHistory = np.zeros((self.numIntervals,self.numberOfParticles))
        self.alivenessHistory = np.zeros((self.numIntervals,self.numberOfParticles))

        self.hungerLevelHistory[0] = hungerLevels
        self.alivenessHistory[0] = alive

        everyoneDead = False

        if self.colourType == ColourType.EXAMPLE:
            self.exampleId = np.random.choice(self.numberOfParticles, 1)
        for t in range(self.numIntervals):
            self.t = t
            previousHungerLevels = hungerLevels
            

            self.updateFoodEvents()
            # if t % 5000 == 0:
            #     print(f"t={t}/{self.tmax}")
            # if self.t % 100 == 0:
            #     print(f"{t}: {ServiceMetric.computeGlobalOrder(orientations)}")

            # all neighbours (including self)
            neighbours = ServiceVicsekHelper.getNeighboursWithLimitedVision(positions=positions, orientations=orientations, domainSize=self.domainSize,
                                                                            radius=self.radius, degreesOfVision=self.degreesOfVision)
            neighbours = neighbours * np.array([alive]*self.numberOfParticles)
            stressLevels = self.updateStressLevels(stressLevels, neighbours)
            orientations, nsms, ks, speeds, blocked, self.colours = self.handleEvents(t, positions, orientations, nsms, ks, speeds, activationTimeDelays)
            self.colours = np.where(alive, 'k', 'w')

            speeds, hungerLevels = self.handleFoodEvents(t, positions, speeds, hungerLevels)

            if self.switchSummary != None:
                thresholdEvaluationChoiceValues = ServiceThresholdEvaluation.getThresholdEvaluationValuesForChoice(thresholdEvaluationMethod=self.thresholdEvaluationMethod, positions=positions, orientations=orientations, neighbours=neighbours, domainSize=self.domainSize)

                self.thresholdEvaluationChoiceValuesHistory.append(thresholdEvaluationChoiceValues)
            
                if SwitchType.NEIGHBOUR_SELECTION_MECHANISM in self.switchSummary.switches.keys():
                    nsms = self.getDecisions(t, neighbours, thresholdEvaluationChoiceValues, self.thresholdEvaluationChoiceValuesHistory, SwitchType.NEIGHBOUR_SELECTION_MECHANISM, nsms, blocked, stressLevels)
                if SwitchType.K in self.switchSummary.switches.keys():
                    ks = self.getDecisions(t, neighbours, thresholdEvaluationChoiceValues, self.thresholdEvaluationChoiceValuesHistory, SwitchType.K, ks, blocked, stressLevels)
                if SwitchType.SPEED in self.switchSummary.switches.keys():
                    speeds = self.getDecisions(t, neighbours, thresholdEvaluationChoiceValues, self.thresholdEvaluationChoiceValuesHistory, SwitchType.SPEED, speeds, blocked, stressLevels)
                if SwitchType.ACTIVATION_TIME_DELAY in self.switchSummary.switches.keys():
                    activationTimeDelays = self.getDecisions(t, neighbours, thresholdEvaluationChoiceValues, self.thresholdEvaluationChoiceValuesHistory, SwitchType.ACTIVATION_TIME_DELAY, activationTimeDelays, blocked, stressLevels)

            orientations = self.computeNewOrientations(neighbours, positions, orientations, nsms, ks, activationTimeDelays)

            positions += self.dt*(orientations.T * speeds).T
            positions += -self.domainSize*np.floor(positions/self.domainSize)

            # if an individual is not feeding, it gets more hungry at every timestep
            reducedHungerLevels = hungerLevels - 0.1
            hungerLevels = np.where(hungerLevels == previousHungerLevels, reducedHungerLevels, hungerLevels)

            alive = np.where(hungerLevels > 0, True, False)

            self.positionsHistory[t,:,:]=positions
            self.orientationsHistory[t,:,:]=orientations
            self.stressLevelsHistory[t,:]=stressLevels
            self.hungerLevelHistory[t,:]=hungerLevels
            self.alivenessHistory[t,:]=alive

            self.appendSwitchValues(nsms, ks, speeds, activationTimeDelays)
            self.coloursHistory[t] = self.colours

            if np.count_nonzero(alive) == 0 and everyoneDead == False:
                everyoneDead = True
                print(f"everyone is dead by timestep {t}")
            
            # if t % 500 == 0:
            #     print(f"t={t}, th={self.thresholdEvaluationMethod.name}, order={ServiceMetric.computeGlobalOrder(orientations)}")
            
        print(f"num foodev: {len(self.foodEvents)}")
        return (self.dt*np.arange(self.numIntervals), self.positionsHistory, self.orientationsHistory), self.switchTypeValuesHistory, self.coloursHistory, self.stressLevelsHistory, self.hungerLevelHistory, self.alivenessHistory, self.foodEvents
