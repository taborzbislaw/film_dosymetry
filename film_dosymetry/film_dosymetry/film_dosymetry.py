import json
import numpy as np
import logging
import os
import signal
import shutil
from typing import Annotated, Optional
import time
import subprocess
import threading

import vtk
import qt

from qt import QObject, Signal, Slot

import inspect

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
    
try:
    from scipy.optimize import minimize
except ModuleNotFoundError:
    slicer.util.pip_install("scipy")
    from scipy.optimize import minimize

try:
    import cv2
except ModuleNotFoundError:
    slicer.util.pip_install("opencv-contrib-python")
    import cv2    

try:
    import SimpleITK as sitk
except ModuleNotFoundError:
    slicer.util.pip_install("SimpleITK")
    import SimpleITK as sitk    

from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLVectorVolumeNode
from slicer import vtkMRMLMarkupsROINode

#
# film_dosymetry
#

def read_json(fname):
    # Load data from JSON file
    with open(fname, 'r') as f:
        data = json.load(f)
    return data

def rational_func(x, a, b, c):
    return (a + b*x) / (c + x)

def getIJKCoordinates(roiNode,volumeNode):
# https://slicer.readthedocs.io/en/latest/developer_guide/script_repository/volumes.html "Get volume voxel coordinates from markup control point RAS coordinates"
    point_Ras = [0, 0, 0]
    roiNode.GetNthControlPointPositionWorld(0, point_Ras)

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


class film_dosymetry(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("film_dosymetry")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Paweł Wołowiec (Holly Cross Cancer Center)", "Zbisław Tabor (AGH University of Cracow)"] 
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#film_dosymetry">module documentation</a>.
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

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

#
# film_dosymetryParameterNode
#


@parameterNodeWrapper
class film_dosymetryParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLVectorVolumeNode
    outputVolume: vtkMRMLScalarVolumeNode
    roi: vtkMRMLMarkupsROINode
    
#
# film_dosymetryWidget
#

# https://wiki.qt.io/Qt_for_Python_Signals_and_Slots
class Communicate(QObject):                                                 
    setValue = Signal(int) 

class film_dosymetryWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        self.calibrationCoefficients = None
        self.monitor = None
        self.processes = None
        self.stop_event = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/film_dosymetry.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        
        self.monitor = Communicate()
        self.monitor.setValue.connect(self.setProgressBar)
        
        self.stop_event = threading.Event()
        
        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = film_dosymetryLogic()

        # Connections
        #self.ui.inputSelector.connect("currentNodeChanged(bool)",self.vectorVolumeChanged)
        #self.ui.outputSelector.connect("currentNodeChanged(bool)",self.scalarVolumeChanged)

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
                
        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    #def vectorVolumeChanged(self) -> None:
    #    slicer.util.setSliceViewerLayers(background = self.ui.inputSelector.currentNode())
    #    slicer.util.resetSliceViews()

    #def scalarVolumeChanged(self) -> None:
    #    slicer.util.setSliceViewerLayers(background = self.ui.outputSelector.currentNode())
    #    slicer.util.resetSliceViews()
        
        
    @Slot(int)                                                                  
    def setProgressBar(self,value) -> None:     
        #workDir = os.path.dirname(__file__)
        #f = open(workDir + '/Resources/worker.txt','a')         
        #print(value,file=f,flush=True)     
        #f.close()
        self.ui.progressBar.setValue(value)
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
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[film_dosymetryParameterNode]) -> None:
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
        if self._parameterNode and self._parameterNode.inputVolume:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False

    def monitorAndColmpeteCalculations(self,img, STARTS, ENDS, COL_START, COL_END) -> None:     
            
        workDir = os.path.dirname(__file__)
       
        old_mean = 0
        while True:
            if self.stop_event.is_set():
                break
            percents = []
            currents = []
            ends = []
            for n in range(self.ui.spinBoxNumberOfWorkers.value):
                fname = workDir + '/Resources/log' + str(n) + '.txt'
                if os.path.isfile(fname):
                    try:
                        f = open(fname,'r')
                        l,s,e = f.readlines()[0].split()
                        f.close()
                        percent = (int(l)-int(s))/(int(e)-int(s))*100
                        percents.append(percent)
                        ends.append(int(e))
                        currents.append(int(l)+1)
                    except:
                        pass
                        
            if len(percents)==self.ui.spinBoxNumberOfWorkers.value:
                f = open(workDir + '/Resources/log.txt','a')
                print(percents,currents,ends,file=f)
                f.close()    
                mean = int(np.mean(percents))
                if mean > old_mean:                    
                    self.monitor.setValue.emit(mean)
                    old_mean = mean
                    
                diff = np.sum([i-j for i,j in zip(ends,currents)])                
                if diff==0:
                    break

        if not self.stop_event.is_set():
            self.monitor.setValue.emit(100)
        
        for n in range(self.ui.spinBoxNumberOfWorkers.value):
            fname = workDir + '/Resources/foo' + str(n) + '.nii.gz'
            flag = 1
            while flag:
                if self.stop_event.is_set():
                    break
                try:
                    sitk.ReadImage(fname)
                    flag = 0
                except:
                    pass
                      
        if not self.stop_event.is_set():
            calibrated_image = np.zeros(img.shape[0:2],dtype=np.float32)
            for n in range(self.ui.spinBoxNumberOfWorkers.value):
                fname = workDir + '/Resources/foo' + str(n) + '.nii.gz'
                imSITK = sitk.ReadImage(fname)
                imgCropped = sitk.GetArrayFromImage(imSITK)
                calibrated_image[STARTS[n]:ENDS[n],COL_START:COL_END] = np.copy(imgCropped[STARTS[n]:ENDS[n],COL_START:COL_END])
                
            calibrated_image = np.reshape((1,) + calibrated_image,calibrated_image.shape)
            nodeName = 'CalibratedFilm'
            calibratedVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", nodeName)
            slicer.util.updateVolumeFromArray(calibratedVolume, calibrated_image)   

            volumeItkToRas = vtk.vtkMatrix4x4()
            self._parameterNode.inputVolume.GetIJKToRASDirectionMatrix(volumeItkToRas)
            calibratedVolume.SetIJKToRASDirectionMatrix(volumeItkToRas)           
            volumeRasToIjk = vtk.vtkMatrix4x4()
            self._parameterNode.inputVolume.GetRASToIJKMatrix(volumeRasToIjk)
            calibratedVolume.SetRASToIJKMatrix(volumeRasToIjk)
            calibratedVolume.SetOrigin(self._parameterNode.inputVolume.GetOrigin())
            calibratedVolume.SetSpacing(self._parameterNode.inputVolume.GetSpacing())
            
            fname = workDir + '/Resources/foo.nii.gz'
            os.remove(fname)
            fname = workDir + '/Resources/coefs.json'
            os.remove(fname)
            fname = workDir + '/Resources/log.txt'
            os.remove(fname)
            for n in range(self.ui.spinBoxNumberOfWorkers.value):
               fname = workDir + '/Resources/foo' + str(n) + '.nii.gz'
               os.remove(fname)
            for n in range(self.ui.spinBoxNumberOfWorkers.value):
               fname = workDir + '/Resources/log' + str(n) + '.txt'
               os.remove(fname)
               
        self.ui.applyButton.setEnabled(True)
        self.processes = None
        self.stop_event.clear()     


    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        
        path = self.ui.CalibrationCoefficientsFilePath.currentPath
        
        if os.path.isfile(path):
            
            self.stop_event.clear()     

            img = slicer.util.arrayFromVolume(self._parameterNode.inputVolume)[0]
            
            spacing = self._parameterNode.inputVolume.GetSpacing()

            if self.ui.checkBoxMedianFilter.isChecked():
                img = cv2.medianBlur(img, ksize=self.ui.spinBoxMedianFilterKernelSize.value)
            
            slicerDir = os.getcwd()
            workDir = os.path.dirname(__file__)
            
            imSITK = sitk.GetImageFromArray(img)
            fname = workDir + '/Resources/foo.nii.gz'
            sitk.WriteImage(imSITK,fname)
            fname = workDir + '/Resources/coefs.json'
            shutil.copy(path,fname)

            ROW_START = 0
            ROW_END = img.shape[0]
            COL_START = 0
            COL_END = img.shape[1]
            
            if self._parameterNode.roi is not None:
                col,row,_ = getIJKCoordinates(self._parameterNode.roi,self._parameterNode.inputVolume)
                size = self._parameterNode.roi.GetSize()
                spacing = self._parameterNode.inputVolume.GetSpacing()
                img = slicer.util.arrayFromVolume(self._parameterNode.inputVolume)[0]
                colSpan,rowSpan,_ = [int(s/sp) for s,sp in zip(size,spacing)]
                ROW_START = int(row - rowSpan//2)
                ROW_END = int(ROW_START + rowSpan)
                COL_START = int(col - colSpan//2)
                COL_END = int(COL_START + colSpan)
            
            STEP = (ROW_END - ROW_START)//self.ui.spinBoxNumberOfWorkers.value
            STARTS = [ROW_START + i*STEP for i in range(self.ui.spinBoxNumberOfWorkers.value)]
            ENDS = [ROW_START + (i+1)*STEP for i in range(self.ui.spinBoxNumberOfWorkers.value)]
            ENDS[-1] = ROW_END
            
            #STEP = 10
            #STARTS = [350 + i*STEP for i in range(self.ui.spinBoxNumberOfWorkers.value)]
            #ENDS = [350 + (i+1)*STEP for i in range(self.ui.spinBoxNumberOfWorkers.value)]
          
#https://github.com/pieper/SlicerParallelProcessing/blob/master/Processes/Processes.py
            self.ui.progressBar.value = 0
            self.processes = []
            pids = []
            for n in range(self.ui.spinBoxNumberOfWorkers.value):
                process = subprocess.Popen([slicerDir + "/bin/PythonSlicer", workDir + '/optimize.py',str(n),
                          str(STARTS[n]),str(ENDS[n]),str(COL_START),str(COL_END),str(self.ui.doubleSpinBoxTolerance.value),str(self.ui.spinBoxIterations.value),
                          str(self.ui.doubleSpinBoxNormalization.value),str(self.ui.doubleSpinBoxMaxDose.value)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,preexec_fn=os.setpgrp)
                self.processes.append(process.pid)
                pids.append(process.pid)
            
#            workDir = os.path.dirname(__file__)
#            f = open(workDir + '/Resources/pids.txt','w')
#            print(pids,file=f)
#            f.close()
            
            self.ui.applyButton.setEnabled(False)
            self.ui.progressBar.setValue(0)

            self.monitorAndColmpeteCalculations(img, STARTS, ENDS, COL_START, COL_END)

            
#
# film_dosymetryLogic
#


class film_dosymetryLogic(ScriptedLoadableModuleLogic):
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
        return film_dosymetryParameterNode(super().getParameterNode())

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            "InputVolume": inputVolume.GetID(),
            "OutputVolume": outputVolume.GetID(),
            "ThresholdValue": imageThreshold,
            "ThresholdType": "Above" if invert else "Below",
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")


#
# film_dosymetryTest
#


class film_dosymetryTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_film_dosymetry1()

    def test_film_dosymetry1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("film_dosymetry1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = film_dosymetryLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
