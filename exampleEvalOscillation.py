import numpy as np

from enums.EnumMetrics import Metrics
from evaluators.EvaluatorAvalanches import EvaluatorAvalanches
from evaluators.EvaluatorMultiComp import EvaluatorMultiAvgComp
import services.ServiceSavedModel as ssm

for noisePercentage in [1,2,3,4]:
    for use_single_speed in [True, False]:
        for occlusion_active in [True, False]:
            for fov in [0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi]:
                filename = f"test_singlespeed={use_single_speed}_occl={occlusion_active}_fov={fov}_noiseP={noisePercentage}_d=0.03_r=10_tmax=10000_1"
                modelParams, simulationData = ssm.loadModels([f"{filename}.json"], loadSwitchValues=False)
                times, positions, orientations = simulationData[0]

                evaluator = EvaluatorAvalanches(orientations=orientations, orderThreshold=0.9, savePath=f"avalanches_{filename}", show=False)
                evaluator.evaluateAvalanches()

                evaluator = EvaluatorMultiAvgComp(modelParams=modelParams, metric=Metrics.ORDER, simulationData=simulationData, evaluationTimestepInterval=1)
                evaluator.evaluateAndVisualize(labels=[''], xLabel='timesteps', yLabel='order', savePath=f"order_{filename}.jpeg")