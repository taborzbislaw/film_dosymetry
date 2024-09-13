import logging
import os
import numpy as np
import json

try:
    import cv2
except ModuleNotFoundError:
  slicer.util.pip_install("opencv-contrib-python")
  import cv2

try:
    from scipy.optimize import curve_fit
except ModuleNotFoundError:
    slicer.util.pip_install("scipy")
    from scipy.optimize import curve_fit
    
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  
    slicer.util.pip_install("matplotlib")
    import matplotlib.pyplot as plt

from typing import Annotated, Optional

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLVectorVolumeNode
from slicer import vtkMRMLMarkupsFiducialNode

#
# film_calibration
#

def rational_func(x, a, b, c):
    return (a + b*x) / (c + x)


class film_calibration(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("film_calibration")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Paweł Wołowiec (Holly Cross Cancer Center)","Zbisław Tabor (AGH University of Cracow)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#film_calibration">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

#
# film_calibrationParameterNode
#


@parameterNodeWrapper
class film_calibrationParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLVectorVolumeNode
    
    markup1: vtkMRMLMarkupsFiducialNode 
    markup2: vtkMRMLMarkupsFiducialNode 
    markup3: vtkMRMLMarkupsFiducialNode 
    markup4: vtkMRMLMarkupsFiducialNode 
    markup5: vtkMRMLMarkupsFiducialNode 
    markup6: vtkMRMLMarkupsFiducialNode 
    markup7: vtkMRMLMarkupsFiducialNode 
    markup8: vtkMRMLMarkupsFiducialNode 
    markup9: vtkMRMLMarkupsFiducialNode 
    markup10: vtkMRMLMarkupsFiducialNode      

#
# film_calibrationWidget
#


class film_calibrationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        
        self._parameterNodeGuiTag = None
        
        self.parameters = None
        self.errors = None
        self.means = None
        self.stds = None
        self.doses = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/film_calibration.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = film_calibrationLogic()
        
        self.ui.markerSelector1.setVisible(False)
        self.ui.doseSelector1.setVisible(False)
        self.ui.label_marker1.setVisible(False)
        self.ui.label_dose1.setVisible(False)

        self.ui.markerSelector2.setVisible(False)
        self.ui.doseSelector2.setVisible(False)
        self.ui.label_marker2.setVisible(False)
        self.ui.label_dose2.setVisible(False)

        self.ui.markerSelector3.setVisible(False)
        self.ui.doseSelector3.setVisible(False)
        self.ui.label_marker3.setVisible(False)
        self.ui.label_dose3.setVisible(False)

        self.ui.markerSelector4.setVisible(False)
        self.ui.doseSelector4.setVisible(False)
        self.ui.label_marker4.setVisible(False)
        self.ui.label_dose4.setVisible(False)

        self.ui.markerSelector5.setVisible(False)
        self.ui.doseSelector5.setVisible(False)
        self.ui.label_marker5.setVisible(False)
        self.ui.label_dose5.setVisible(False)

        self.ui.markerSelector6.setVisible(False)
        self.ui.doseSelector6.setVisible(False)
        self.ui.label_marker6.setVisible(False)
        self.ui.label_dose6.setVisible(False)

        self.ui.markerSelector7.setVisible(False)
        self.ui.doseSelector7.setVisible(False)
        self.ui.label_marker7.setVisible(False)
        self.ui.label_dose7.setVisible(False)

        self.ui.markerSelector8.setVisible(False)
        self.ui.doseSelector8.setVisible(False)
        self.ui.label_marker8.setVisible(False)
        self.ui.label_dose8.setVisible(False)

        self.ui.markerSelector9.setVisible(False)
        self.ui.doseSelector9.setVisible(False)
        self.ui.label_marker9.setVisible(False)
        self.ui.label_dose9.setVisible(False)

        self.ui.markerSelector10.setVisible(False)
        self.ui.doseSelector10.setVisible(False)
        self.ui.label_marker10.setVisible(False)
        self.ui.label_dose10.setVisible(False)

        self.ui.applyButton.enabled = False

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.addMarkerButton.connect('clicked(bool)',self.addMarkers)
        self.ui.removeMarkerButton.connect('clicked(bool)',self.removeMarkers)

        self.ui.inputSelector.connect("currentNodeChanged(bool)",self.volumeChanged)
        
        self.ui.saveButton.connect("clicked(bool)",self.saveCalibrationData)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def saveCalibrationData(self) ->None:
        pathCoefs = self.ui.CalibrationCoefficientsFilePath.currentPath
        pathPlot = self.ui.CalibrationPlotFilePath.currentPath 
        
        #workDir = os.path.dirname(__file__)
        #f = open(workDir + '/log.txt','a')
        #print(pathCoefs,pathPlot,file=f)
        #f.close()

        try:
            calib_data = {}
            calib_data['parameters'] = {}
            calib_data['errors'] = {}
            calib_data['parameters']['red'] = list(self.parameters[0])
            calib_data['parameters']['green'] = list(self.parameters[1])
            calib_data['parameters']['blue'] = list(self.parameters[2])
            calib_data['errors']['red'] = list(self.errors[0])
            calib_data['errors']['green'] = list(self.errors[1])
            calib_data['errors']['blue'] = list(self.errors[2])
            
            f=open(pathCoefs,'w')
            print(json.dumps(calib_data, indent=4),file = f)
            f.close()
        except:
            pass
            
        try:
            plt.rcParams['font.family'] = 'sans serif'
            plt.rcParams['font.size'] = 12

            plt.figure(figsize=(10,10))

            plt.errorbar(self.doses,self.means[0],yerr = self.stds[0],fmt='ro',capsize=5)
            plt.errorbar(self.doses,self.means[1],yerr = self.stds[1],fmt='go',capsize=5)
            plt.errorbar(self.doses,self.means[2],yerr = self.stds[2],fmt='bo',capsize=5)

            for channel,color in zip([0,1,2],['r','g','b']):
                y_pred = rational_func(np.array(self.doses),*self.parameters[channel])
                plt.plot(self.doses,y_pred,color)

            plt.xlabel('dose[cGy]')
            plt.ylabel('mean gray level')
            plt.savefig(pathPlot)
        except:
            pass
         
    def addMarkers(self) -> None:
        if self.ui.markerSelector1.isVisible() == False:
            self.ui.markerSelector1.setVisible(True)
            self.ui.doseSelector1.setVisible(True)
            self.ui.label_marker1.setVisible(True)
            self.ui.label_dose1.setVisible(True)
            if not self._parameterNode.markup1:
                markup = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
                if markup:
                    self._parameterNode.markup1 = markup
            return

        if self.ui.markerSelector2.isVisible() == False:
            self.ui.markerSelector2.setVisible(True)
            self.ui.doseSelector2.setVisible(True)
            self.ui.label_marker2.setVisible(True)
            self.ui.label_dose2.setVisible(True)
            if not self._parameterNode.markup2:
                markup = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
                if markup:
                    self._parameterNode.markup2 = markup
            return

        if self.ui.markerSelector3.isVisible() == False:
            self.ui.markerSelector3.setVisible(True)
            self.ui.doseSelector3.setVisible(True)
            self.ui.label_marker3.setVisible(True)
            self.ui.label_dose3.setVisible(True)
            if not self._parameterNode.markup3:
                markup = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
                if markup:
                    self._parameterNode.markup3 = markup
            return

        if self.ui.markerSelector4.isVisible() == False:
            self.ui.markerSelector4.setVisible(True)
            self.ui.doseSelector4.setVisible(True)
            self.ui.label_marker4.setVisible(True)
            self.ui.label_dose4.setVisible(True)
            if not self._parameterNode.markup4:
                markup = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
                if markup:
                    self._parameterNode.markup4 = markup
            return

        if self.ui.markerSelector5.isVisible() == False:
            self.ui.markerSelector5.setVisible(True)
            self.ui.doseSelector5.setVisible(True)
            self.ui.label_marker5.setVisible(True)
            self.ui.label_dose5.setVisible(True)
            if not self._parameterNode.markup5:
                markup = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
                if markup:
                    self._parameterNode.markup5 = markup
            return

        if self.ui.markerSelector6.isVisible() == False:
            self.ui.markerSelector6.setVisible(True)
            self.ui.doseSelector6.setVisible(True)
            self.ui.label_marker6.setVisible(True)
            self.ui.label_dose6.setVisible(True)
            if not self._parameterNode.markup6:
                markup = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
                if markup:
                    self._parameterNode.markup6 = markup
            return

        if self.ui.markerSelector7.isVisible() == False:
            self.ui.markerSelector7.setVisible(True)
            self.ui.doseSelector7.setVisible(True)
            self.ui.label_marker7.setVisible(True)
            self.ui.label_dose7.setVisible(True)
            if not self._parameterNode.markup7:
                markup = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
                if markup:
                    self._parameterNode.markup7 = markup
            return

        if self.ui.markerSelector8.isVisible() == False:
            self.ui.markerSelector8.setVisible(True)
            self.ui.doseSelector8.setVisible(True)
            self.ui.label_marker8.setVisible(True)
            self.ui.label_dose8.setVisible(True)
            if not self._parameterNode.markup8:
                markup = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
                if markup:
                    self._parameterNode.markup8 = markup
            return

        if self.ui.markerSelector9.isVisible() == False:
            self.ui.markerSelector9.setVisible(True)
            self.ui.doseSelector9.setVisible(True)
            self.ui.label_marker9.setVisible(True)
            self.ui.label_dose9.setVisible(True)
            if not self._parameterNode.markup9:
                markup = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
                if markup:
                    self._parameterNode.markup9 = markup
            return

        if self.ui.markerSelector10.isVisible() == False:
            self.ui.markerSelector10.setVisible(True)
            self.ui.doseSelector10.setVisible(True)
            self.ui.label_marker10.setVisible(True)
            self.ui.label_dose10.setVisible(True)
            if not self._parameterNode.markup10:
                markup = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
                if markup:
                    self._parameterNode.markup10 = markup
            return

    def removeMarkers(self) -> None:
        if self.ui.markerSelector10.isVisible() == True:
            self.ui.markerSelector10.setVisible(False)
            self.ui.doseSelector10.setVisible(False)
            self.ui.label_marker10.setVisible(False)
            self.ui.label_dose10.setVisible(False)
            if self._parameterNode.markup10:
                self._parameterNode.markup10 = None
            return

        if self.ui.markerSelector9.isVisible() == True:
            self.ui.markerSelector9.setVisible(False)
            self.ui.doseSelector9.setVisible(False)
            self.ui.label_marker9.setVisible(False)
            self.ui.label_dose9.setVisible(False)
            if self._parameterNode.markup9:
                self._parameterNode.markup9 = None
            return

        if self.ui.markerSelector8.isVisible() == True:
            self.ui.markerSelector8.setVisible(False)
            self.ui.doseSelector8.setVisible(False)
            self.ui.label_marker8.setVisible(False)
            self.ui.label_dose8.setVisible(False)
            if self._parameterNode.markup8:
                self._parameterNode.markup8 = None
            return

        if self.ui.markerSelector7.isVisible() == True:
            self.ui.markerSelector7.setVisible(False)
            self.ui.doseSelector7.setVisible(False)
            self.ui.label_marker7.setVisible(False)
            self.ui.label_dose7.setVisible(False)
            if self._parameterNode.markup7:
                self._parameterNode.markup7 = None
            return

        if self.ui.markerSelector6.isVisible() == True:
            self.ui.markerSelector6.setVisible(False)
            self.ui.doseSelector6.setVisible(False)
            self.ui.label_marker6.setVisible(False)
            self.ui.label_dose6.setVisible(False)
            if self._parameterNode.markup6:
                self._parameterNode.markup6 = None
            return

        if self.ui.markerSelector5.isVisible() == True:
            self.ui.markerSelector5.setVisible(False)
            self.ui.doseSelector5.setVisible(False)
            self.ui.label_marker5.setVisible(False)
            self.ui.label_dose5.setVisible(False)
            if self._parameterNode.markup5:
                self._parameterNode.markup5 = None
            return

        if self.ui.markerSelector4.isVisible() == True:
            self.ui.markerSelector4.setVisible(False)
            self.ui.doseSelector4.setVisible(False)
            self.ui.label_marker4.setVisible(False)
            self.ui.label_dose4.setVisible(False)
            if self._parameterNode.markup4:
                self._parameterNode.markup4 = None
            return

        if self.ui.markerSelector3.isVisible() == True:
            self.ui.markerSelector3.setVisible(False)
            self.ui.doseSelector3.setVisible(False)
            self.ui.label_marker3.setVisible(False)
            self.ui.label_dose3.setVisible(False)
            if self._parameterNode.markup3:
                self._parameterNode.markup3 = None
            return

        if self.ui.markerSelector2.isVisible() == True:
            self.ui.markerSelector2.setVisible(False)
            self.ui.doseSelector2.setVisible(False)
            self.ui.label_marker2.setVisible(False)
            self.ui.label_dose2.setVisible(False)
            if self._parameterNode.markup2:
                self._parameterNode.markup2 = None
            return

        if self.ui.markerSelector1.isVisible() == True:
            self.ui.markerSelector1.setVisible(False)
            self.ui.doseSelector1.setVisible(False)
            self.ui.label_marker1.setVisible(False)
            self.ui.label_dose1.setVisible(False)
            if self._parameterNode.markup1:
                self._parameterNode.markup1 = None
            return
 
    def volumeChanged(self,flag) -> None:
        slicer.util.setSliceViewerLayers(background = self.ui.inputSelector.currentNode())
        slicer.util.resetSliceViews()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLVectorVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[film_calibrationParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.markup4:
            self.ui.applyButton.toolTip = _("Run calibration")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input volume node")
            self.ui.applyButton.enabled = False

        
    def onApplyButton(self) -> None:
        
        def getIJKCoordinates(pointNode,volumeNode):
     # https://slicer.readthedocs.io/en/latest/developer_guide/script_repository/volumes.html "Get volume voxel coordinates from markup control point RAS coordinates"
            point_Ras = [0, 0, 0]
            pointNode.GetNthControlPointPositionWorld(0, point_Ras)
            
            transformRasToVolumeRas = vtk.vtkGeneralTransform()
            slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, volumeNode.GetParentTransformNode(), transformRasToVolumeRas)
            point_VolumeRas = transformRasToVolumeRas.TransformPoint(point_Ras)

            # Get voxel coordinates from physical coordinates
            volumeRasToIjk = vtk.vtkMatrix4x4()
            volumeNode.GetRASToIJKMatrix(volumeRasToIjk)
            point_Ijk = [0, 0, 0, 1]
            volumeRasToIjk.MultiplyPoint(np.append(point_VolumeRas,1.0), point_Ijk)
            point_Ijk = [ int(round(c)) for c in point_Ijk[0:3] ]    

            return point_Ijk

        img = slicer.util.arrayFromVolume(self._parameterNode.inputVolume)[0]
        
        if self.ui.applyMedianFilterCheckBox.isChecked():
            img = cv2.medianBlur(imArr, ksize=self.ui.spinBoxKernelSize.value)

        pointNodes = []
        doses = []
        if self.ui.markerSelector1.isVisible():
            pointNodes.append(self._parameterNode.markup1)
            doses.append(self.ui.doseSelector1.value)
        if self.ui.markerSelector2.isVisible():
            pointNodes.append(self._parameterNode.markup2)
            doses.append(self.ui.doseSelector2.value)
        if self.ui.markerSelector3.isVisible():
            pointNodes.append(self._parameterNode.markup3)
            doses.append(self.ui.doseSelector3.value)
        if self.ui.markerSelector4.isVisible():
            pointNodes.append(self._parameterNode.markup4)
            doses.append(self.ui.doseSelector4.value)
        if self.ui.markerSelector5.isVisible():
            pointNodes.append(self._parameterNode.markup5)
            doses.append(self.ui.doseSelector5.value)
        if self.ui.markerSelector6.isVisible():
            pointNodes.append(self._parameterNode.markup6)
            doses.append(self.ui.doseSelector6.value)
        if self.ui.markerSelector7.isVisible():
            pointNodes.append(self._parameterNode.markup7)
            doses.append(self.ui.doseSelector7.value)
        if self.ui.markerSelector8.isVisible():
            pointNodes.append(self._parameterNode.markup8)
            doses.append(self.ui.doseSelector8.value)
        if self.ui.markerSelector9.isVisible():
            pointNodes.append(self._parameterNode.markup9)
            doses.append(self.ui.doseSelector9.value)
        if self.ui.markerSelector10.isVisible():
            pointNodes.append(self._parameterNode.markup10)
            doses.append(self.ui.doseSelector10.value)
        
        pointsITK = []
        for p in pointNodes:
            pointsITK.append(getIJKCoordinates(p,self._parameterNode.inputVolume))
        
        BOX = self.ui.boxSizeSpinBox.value
        NORM_FACTOR = self.ui.doubleSpinBoxNormalization.value
        means = [[np.mean(img[p[1]-BOX:p[1]+BOX,p[0]-BOX:p[0]+BOX,channel]/NORM_FACTOR) for p in pointsITK] for channel in [0,1,2]]
        stds =  [[np.std(img[p[1]-BOX:p[1]+BOX,p[0]-BOX:p[0]+BOX,channel]/NORM_FACTOR) for p in pointsITK] for channel in [0,1,2]]


        initial_values = [[self.ui.doubleSpinBoxRed1.value,self.ui.doubleSpinBoxRed2.value,self.ui.doubleSpinBoxRed2.value],
                            [self.ui.doubleSpinBoxGreen1.value,self.ui.doubleSpinBoxGreen2.value,self.ui.doubleSpinBoxGreen3.value],
                            [self.ui.doubleSpinBoxBlue1.value,self.ui.doubleSpinBoxBlue2.value,self.ui.doubleSpinBoxBlue3.value]]
        self.parameters = []
        self.errors = []
        for channel,initial in zip([0,1,2], initial_values):
            sigma = 1.0/np.array(stds[channel])**2
            popt, pcov = curve_fit(rational_func, doses, means[channel], p0=initial,sigma=sigma) 
            perr = np.sqrt(np.diag(pcov))
            
            self.parameters.append(popt)
            self.errors.append(perr)

        self.means = means
        self.stds = stds
        self.doses = doses
       
# https://apidocs.slicer.org/main/classvtkMRMLPlotSeriesNode.html#acfadd1795b69c8f7178693c7a4a47ef6

        results = (np.array(means[0]),np.array(doses))
        tableNodeMin=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
        slicer.util.updateTableFromArray(tableNodeMin, results)
        tableNodeMin.GetTable().GetColumn(0).SetName("Red signal")
        tableNodeMin.GetTable().GetColumn(1).SetName("Dose [cGy]")
        
        # Create plot
        plotSeriesNodeMin = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "Red signal")
        plotSeriesNodeMin.SetAndObserveTableNodeID(tableNodeMin.GetID())
        plotSeriesNodeMin.SetXColumnName("Dose [cGy]")
        plotSeriesNodeMin.SetYColumnName("Red signal")
        plotSeriesNodeMin.SetLineStyle(plotSeriesNodeMin.LineStyleNone )
        plotSeriesNodeMin.SetPlotType(plotSeriesNodeMin.PlotTypeScatter )
        plotSeriesNodeMin.SetColor(1.0, 0, 0.0)

        results = (np.array(means[1]),np.array(doses))
        tableNodeMax=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
        slicer.util.updateTableFromArray(tableNodeMax, results)
        tableNodeMax.GetTable().GetColumn(0).SetName("Green signal")
        tableNodeMax.GetTable().GetColumn(1).SetName("Dose [cGy]")
        
        # Create plot
        plotSeriesNodeMax = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "Green signal")
        plotSeriesNodeMax.SetAndObserveTableNodeID(tableNodeMax.GetID())
        plotSeriesNodeMax.SetXColumnName("Dose [cGy]")
        plotSeriesNodeMax.SetYColumnName("Green signal")
        plotSeriesNodeMax.SetLineStyle(plotSeriesNodeMax.LineStyleNone )
        plotSeriesNodeMax.SetPlotType(plotSeriesNodeMax.PlotTypeScatter )
        plotSeriesNodeMax.SetColor(0, 1.0, 0)

        results = (np.array(means[2]),np.array(doses))
        tableNode=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
        slicer.util.updateTableFromArray(tableNode, results)
        tableNode.GetTable().GetColumn(0).SetName("Blue signal")
        tableNode.GetTable().GetColumn(1).SetName("Dose [cGy]")
        
        # Create plot
        plotSeriesNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "Blue signal")
        plotSeriesNode.SetAndObserveTableNodeID(tableNode.GetID())
        plotSeriesNode.SetXColumnName("Dose [cGy]")
        plotSeriesNode.SetYColumnName("Blue signal")
        plotSeriesNode.SetLineStyle(plotSeriesNode.LineStyleNone )
        plotSeriesNode.SetPlotType(plotSeriesNode.PlotTypeScatter )
        plotSeriesNode.SetColor(0, 0.0, 1.0)

        y_pred = rational_func(np.array(self.doses),*self.parameters[0])
        results = (np.array(y_pred),np.array(doses))
        table1=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
        slicer.util.updateTableFromArray(table1, results)
        table1.GetTable().GetColumn(0).SetName("Predicted red signal")
        table1.GetTable().GetColumn(1).SetName("Dose [cGy]")
        
        # Create plot
        plot1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "Predicted red signal")
        plot1.SetAndObserveTableNodeID(table1.GetID())
        plot1.SetXColumnName("Dose [cGy]")
        plot1.SetYColumnName("Predicted red signal")
        plot1.SetMarkerStyle(plot1.MarkerStyleNone )
        plot1.SetPlotType(plot1.PlotTypeScatter )
        plot1.SetColor(1.0, 0.0, 0.0)

        y_pred = rational_func(np.array(self.doses),*self.parameters[1])
        results = (np.array(y_pred),np.array(doses))
        table2=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
        slicer.util.updateTableFromArray(table2, results)
        table2.GetTable().GetColumn(0).SetName("Predicted green signal")
        table2.GetTable().GetColumn(1).SetName("Dose [cGy]")
        
        # Create plot
        plot2 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "Predicted green signal")
        plot2.SetAndObserveTableNodeID(table2.GetID())
        plot2.SetXColumnName("Dose [cGy]")
        plot2.SetYColumnName("Predicted green signal")
        plot2.SetMarkerStyle(plot2.MarkerStyleNone )
        plot2.SetPlotType(plot2.PlotTypeScatter )
        plot2.SetColor(0.0, 1.0, 0.0)

        y_pred = rational_func(np.array(self.doses),*self.parameters[2])
        results = (np.array(y_pred),np.array(doses))
        table3=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
        slicer.util.updateTableFromArray(table3, results)
        table3.GetTable().GetColumn(0).SetName("Predicted blue signal")
        table3.GetTable().GetColumn(1).SetName("Dose [cGy]")
        
        # Create plot
        plot3 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "Predicted blue signal")
        plot3.SetAndObserveTableNodeID(table3.GetID())
        plot3.SetXColumnName("Dose [cGy]")
        plot3.SetYColumnName("Predicted blue signal")
        plot3.SetMarkerStyle(plot3.MarkerStyleNone )
        plot3.SetPlotType(plot3.PlotTypeScatter )
        plot3.SetColor(0.0, 0.0, 1.0)

        # Create chart and add plot
        plotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode")
        plotChartNode.AddAndObservePlotSeriesNodeID(plotSeriesNodeMin.GetID())
        plotChartNode.AddAndObservePlotSeriesNodeID(plotSeriesNodeMax.GetID())
        plotChartNode.AddAndObservePlotSeriesNodeID(plotSeriesNode.GetID())
        
        plotChartNode.AddAndObservePlotSeriesNodeID(plot1.GetID())
        plotChartNode.AddAndObservePlotSeriesNodeID(plot2.GetID())
        plotChartNode.AddAndObservePlotSeriesNodeID(plot3.GetID())

        plotChartNode.SetYAxisTitle('Mean normalized signal')
        plotChartNode.SetXAxisTitle('Dose [cGy]')
        min_dose = np.min(doses) - 0.1*(np.max(doses)-np.min(doses))
        max_dose = np.max(doses) + 0.1*(np.max(doses)-np.min(doses))
        plotChartNode.XAxisRangeAutoOff()
        plotChartNode.SetXAxisRange(min_dose,max_dose)
        
        # Show plot in layout
        slicer.modules.plots.logic().ShowChartInLayout(plotChartNode)


#
# film_calibrationLogic
#


class film_calibrationLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return film_calibrationParameterNode(super().getParameterNode())

    # def process(self,
                # inputVolume: vtkMRMLScalarVolumeNode,
                # outputVolume: vtkMRMLScalarVolumeNode,
                # imageThreshold: float,
                # invert: bool = False,
                # showResult: bool = True) -> None:
        # """
        # Run the processing algorithm.
        # Can be used without GUI widget.
        # :param inputVolume: volume to be thresholded
        # :param outputVolume: thresholding result
        # :param imageThreshold: values above/below this threshold will be set to 0
        # :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        # :param showResult: show output volume in slice viewers
        # """

        # if not inputVolume or not outputVolume:
            # raise ValueError("Input or output volume is invalid")

        # import time

        # startTime = time.time()
        # logging.info("Processing started")

        # # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        # cliParams = {
            # "InputVolume": inputVolume.GetID(),
            # "OutputVolume": outputVolume.GetID(),
            # "ThresholdValue": imageThreshold,
            # "ThresholdType": "Above" if invert else "Below",
        # }
        # cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        # slicer.mrmlScene.RemoveNode(cliNode)

        # stopTime = time.time()
        # logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")


# #
# # film_calibrationTest
# #


# class film_calibrationTest(ScriptedLoadableModuleTest):
    # """
    # This is the test case for your scripted module.
    # Uses ScriptedLoadableModuleTest base class, available at:
    # https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    # """

    # def setUp(self):
        # """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        # slicer.mrmlScene.Clear()

    # def runTest(self):
        # """Run as few or as many tests as needed here."""
        # self.setUp()
        # self.test_film_calibration1()

    # def test_film_calibration1(self):
        # """Ideally you should have several levels of tests.  At the lowest level
        # tests should exercise the functionality of the logic with different inputs
        # (both valid and invalid).  At higher levels your tests should emulate the
        # way the user would interact with your code and confirm that it still works
        # the way you intended.
        # One of the most important features of the tests is that it should alert other
        # developers when their changes will have an impact on the behavior of your
        # module.  For example, if a developer removes a feature that you depend on,
        # your test should break so they know that the feature is needed.
        # """

        # self.delayDisplay("Starting the test")

        # # Get/create input data

        # import SampleData

        # registerSampleData()
        # inputVolume = SampleData.downloadSample("film_calibration1")
        # self.delayDisplay("Loaded test data set")

        # inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(inputScalarRange[0], 0)
        # self.assertEqual(inputScalarRange[1], 695)

        # outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        # threshold = 100

        # # Test the module logic

        # logic = film_calibrationLogic()

        # # Test algorithm with non-inverted threshold
        # logic.process(inputVolume, outputVolume, threshold, True)
        # outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        # self.assertEqual(outputScalarRange[1], threshold)

        # # Test algorithm with inverted threshold
        # logic.process(inputVolume, outputVolume, threshold, False)
        # outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        # self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        # self.delayDisplay("Test passed")
