from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


class Animator(object):
    """
    Animates the quiver plot for the Vicsek data
    """
    def prepareAnimation(self, matplotlibFigure, frames=100, frameInterval = 10):
        """
        Prepares the 2D animator object for animation.

        Parameters:
            - matplotlibFigure (Figure): Matplotlibs figure object.
            - frameInterval (int): The interval between two frames.
            - frames (int): The number of frames used in the animation.

        Returns: 
            self
        """
        self._matplotlibFigure = matplotlibFigure
        self._frames = frames
        self._frameInterval = frameInterval

        return self

    def setSimulationData(self, simulationData, domainSize, colours=None, redIndices=[], showRadiusForExample=False, foodEvents=[]):
        """
        Sets the simulation data.
        
        Parameters:
            - simulationData (array of arrays): The simulation data array containing times, positions and orientations
            - domainSize (tuple of floats): The tuple that represents the lenghts of the square domain in each dimension.
            - colours (array of arrays) [optional]: Contains the colour for every particle at every timestep. By default, all particles will be shown in black
            - redIndices (list) [optional]: A list containing indices of particles that will be shown in red. Will be ignored if a colour array is passed
            - showRadiusForExample (boolean) [optional]: if an example particle is provided in the modelParams, toggles if the perception radius is drawn in the video

        Returns:
            self
        """        
        self._time, self._positions, self._orientations = simulationData
        self._domainSize = domainSize

        if colours is None: # if the colours are provided, we don't mess with those as they may show an example
            a = np.array(len(self._positions[0]) * ['k'])
            if len(redIndices) > 0:
                a[redIndices] = 'r'
            self._colours = len(self._time) * [a.tolist()]
        else:
            self._colours = colours
        
        self._showRadusForExample = showRadiusForExample
        self.foodEvents = foodEvents

        return self
    
    def setParameters(self, n, k, noise, radius):
        self._n = n
        self._k = k
        self._noise = noise
        self._radius = radius

    def setParams(self, modelParams):
        self._n = 56
        self._k = 1
        self._noise = 1
        self._radius = 20
        self._domainSize = (50,50)
        self._exampleId = None

    def showAnimation(self):
        """
        Shows the animation

        Parameters:
            None

        Returns: 
            self
        """
        self._getAnimation()
        plt.show()
        
        return self
    
    def saveAnimation(self, filename, fpsVar=25, codecVar="avi"):
        """
        Saves the animation. Requires FFMPEG

        Parameters:
            None

        Returns:
            Animator
        """
        print("Saving commenced...")
        animation = self._getAnimation()
        animation.save(filename=filename, writer="ffmpeg")
        print("Saving completed.")
        #plt.close()
        return self
    
    def _getAnimation(self):
        return self.animation if 'animation' in self.__dict__ else self._generateAnimation()

    def _generateAnimation(self):
        """
        Generates the animation.

        Parameters:
            None
        
        Returns
            animation object
        """
        self.animation = FuncAnimation(self._matplotlibFigure, self._animate, interval=self._frameInterval, frames = self._frames)

        return self.animation
