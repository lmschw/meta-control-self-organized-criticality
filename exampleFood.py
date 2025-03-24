import time
import numpy as np

from model.VicsekIndividualsMultiSwitchOscillationFood import VicsekWithNeighbourSelectionOscillationFood
from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumSwitchType import SwitchType
from events.ExternalStimulusEvent import ExternalStimulusOrientationChangeEvent
from enums.EnumEventEffect import EventEffect
from enums.EnumDistributionType import DistributionType
from enums.EnumEventSelectionType import EventSelectionType
from enums.EnumColourType import ColourType
from enums.EnumThresholdEvaluationMethod import ThresholdEvaluationMethod

from model.SwitchInformation import SwitchInformation
from model.SwitchSummary import SwitchSummary

import services.ServicePreparation as ServicePreparation
import services.ServiceGeneral as ServiceGeneral
import services.ServiceSavedModel as ServiceSavedModel

from animator.Animator import Animator
from animator.Animator2D import Animator2D
from animator.AnimatorMatplotlib import MatplotlibAnimator

print(ServicePreparation.getRadiusToSeeOnAverageNNeighbours(5, 0.03))

domainSize = (22.36, 22.36)
domainSize = (25, 25)
#domainSize = (50, 50)
noisePercentage = 1
noise = ServicePreparation.getNoiseAmplitudeValueForPercentage(noisePercentage)
print(f"noiseP={noisePercentage}, noise={noise}")
#noise = 0
n = ServicePreparation.getNumberOfParticlesForConstantDensity(0.05, domainSize)
print(n)
speed = 1

threshold = 0.1
numberOfPreviousSteps = 100

radius = 5
k = 1
nsm = NeighbourSelectionMechanism.NEAREST

infoNsm = SwitchInformation(switchType=SwitchType.NEIGHBOUR_SELECTION_MECHANISM, 
                            values=(NeighbourSelectionMechanism.FARTHEST, NeighbourSelectionMechanism.NEAREST),
                            thresholds=[threshold],
                            numberPreviousStepsForThreshold=numberOfPreviousSteps
                            )

infoK = SwitchInformation(switchType=SwitchType.K, 
                        values=(5, 1),
                        thresholds=[threshold],
                        numberPreviousStepsForThreshold=numberOfPreviousSteps
                        )

infoSpeed = SwitchInformation(switchType=SwitchType.SPEED, 
                        values=(1, 0.1),
                        thresholds=[threshold],
                        numberPreviousStepsForThreshold=numberOfPreviousSteps
                        )

switchSummary = SwitchSummary([infoNsm])

"""
switchType = SwitchType.NEIGHBOUR_SELECTION_MECHANISM
switchValues = (NeighbourSelectionMechanism.FARTHEST,NeighbourSelectionMechanism.NEAREST)
"""

"""
switchType = SwitchType.K
switchValues = (5,1)
"""
"""
switchType = SwitchType.SPEED
switchValues = (0.1, 1)
"""

tmax = 10000

threshold = [threshold]

tstart = time.time()

ServiceGeneral.logWithTime("start")
stress_num_neighbours = 9
stress_delta = 0.05
stress_delta = 0

foodAmount = 10
maxFood = 50
foodAppearanceProbability = 0.02

for i in range(1, 2):
    initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n, angleX=0.5, angleY=0.5)
    simulator = VicsekWithNeighbourSelectionOscillationFood(domainSize=domainSize,
                                            radius=radius,
                                            noise=noise,
                                            numberOfParticles=n,
                                            k=k,
                                            neighbourSelectionMechanism=nsm,
                                            speed=speed,
                                            switchSummary=switchSummary,
                                            degreesOfVision=np.pi*2,
                                            maxFood=maxFood,
                                            foodAppearanceProbability=foodAppearanceProbability,
                                            foodSourceAmount=foodAmount,
                                            foodEvents=[],
                                            events=[],
                                            colourType=None,
                                            thresholdEvaluationMethod=ThresholdEvaluationMethod.LOCAL_ORDER,
                                            updateIfNoNeighbours=True,
                                            stress_num_neighbours=stress_num_neighbours,
                                            social_stress_delta=stress_delta,
                                            individualistic_stress_delta=stress_delta)
    simulationData, switchTypeValues, colours, stressLevels, hungerLevels, alive, foodEvents = simulator.simulate(initialState=initialState, tmax=tmax)

    print(alive[-1])
    print(colours[-1])
    # Initalise the animator
    animator = MatplotlibAnimator(simulationData, (domainSize[0],domainSize[1],100), colours, foodEvents=foodEvents)

    # prepare the animator
    summary = simulator.getParameterSummary()
    preparedAnimator = animator.prepare(Animator2D(summary), frames=tmax)
    preparedAnimator.setParams(summary)

    filename = f"test"
    preparedAnimator.saveAnimation(f"{filename}.mp4")


    # ServiceSavedModel.saveModel(simulationData=simulationData, path=f"test_stress_{stress_num_neighbours}_tmax={tmax}_{i}.json", 
    #                             modelParams=simulator.getParameterSummary(), switchValues=switchTypeValues, colours=colours, 
    #                             stressLevels=stressLevels, hungerLevels=hungerLevels, alive=alive, foodEvents=foodEvents)

tend = time.time()
ServiceGeneral.logWithTime(f"duration: {ServiceGeneral.formatTime(tend-tstart)}")