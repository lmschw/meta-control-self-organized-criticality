import time
import numpy as np

from animator.AnimatorMatplotlib import MatplotlibAnimator
from animator.Animator2D import Animator2D
from model.Vicsek import VicsekWithNeighbourSelection
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

print(ServicePreparation.getRadiusToSeeOnAverageNNeighbours(5, 0.03))

domainSize = (22.36, 22.36)
domainSize = (25, 25)
#domainSize = (50, 50)
#noise = 0
density = 0.03
n = ServicePreparation.getNumberOfParticlesForConstantDensity(density, domainSize)
print(n)
speed = 0.5

threshold = 0.1
numberOfPreviousSteps = 100

radius = 10
k = 1
nsm = NeighbourSelectionMechanism.NEAREST


tmax = 10000

threshold = [threshold]

tstart = time.time()

ServiceGeneral.logWithTime("start")

noisePercentage = 1
noise = ServicePreparation.getNoiseAmplitudeValueForPercentage(noisePercentage)
print(f"noiseP={noisePercentage}, noise={noise}")

use_single_speed = True
occlusion_active = False
fov = 2*np.pi

for noisePercentage in [1,2,3,4]:
    for use_single_speed in [True, False]:
        for vary_speed in [True, False]:
            for occlusion_active in [True, False]:
                for fov in [0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi]:
                    for i in range(1, 2):
                        ServiceGeneral.logWithTime(f"noise={noisePercentage}, ss={use_single_speed}, occ={occlusion_active}, fov={fov}, i={i}")
                        initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n, angleX=0.5, angleY=0.5)
                        simulator = VicsekWithNeighbourSelection(domainSize=domainSize,
                                                                radius=radius,
                                                                noise=noise,
                                                                numberOfParticles=n,
                                                                speed=speed,
                                                                use_single_speed=use_single_speed,
                                                                vary_speed_throughout=vary_speed,
                                                                degreesOfVision=fov,
                                                                occlusion_active=occlusion_active,
                                                                returnHistories=True)
                        simulationData = simulator.simulate(initialState=initialState, tmax=tmax)
                        #simulationData, switchTypeValues, colours = simulator.simulate(tmax=tmax)
                        import services.ServiceMetric as sm
                        times, positions, orientations = simulationData
                        print("order at end:")
                        print(sm.computeGlobalOrder(orientations[-1]))

                        ServiceSavedModel.saveModel(simulationData=simulationData, path=f"test_singlespeed={use_single_speed}_vary={vary_speed}_occl={occlusion_active}_fov={fov}_noiseP={noisePercentage}_d={density}_r={radius}_tmax={tmax}_{i}.json", 
                                                    modelParams=simulator.getParameterSummary())
                        
                        """
                        animator = MatplotlibAnimator(simulationData, (domainSize[0],domainSize[1],100))

                        # prepare the animator
                        summary = simulator.getParameterSummary()
                        preparedAnimator = animator.prepare(Animator2D(summary), frames=tmax)
                        preparedAnimator.setParams(summary)

                        filename = f"test"
                        preparedAnimator.saveAnimation(f"{filename}.mp4")
                        """


tend = time.time()
ServiceGeneral.logWithTime(f"duration: {ServiceGeneral.formatTime(tend-tstart)}")