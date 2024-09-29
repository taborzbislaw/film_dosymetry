import logging
import os
import numpy as np
from typing import Annotated, Optional
import math

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

from slicer import vtkMRMLScalarVolumeNode, vtkMRMLMarkupsROINode

try:
    import SimpleITK as sitk
except ModuleNotFoundError:
    slicer.util.pip_install("SimpleITK")
    import SimpleITK as sitk    

try:
    from pycimg import CImg
except ModuleNotFoundError:
    slicer.util.pip_install("pycimg")
    from pycimg import CImg

try:
    import pydicom
except ModuleNotFoundError:
    slicer.util.pip_install("pydicom")
    import pydicom

#try:
#    import pymedphys
#except:
#    slicer.util.pip_install("pymedphys")
#    import pymedphys
#
###############################################################################    
###############################################################################    
###############################################################################    

def getIJKCoordinates1(X,Y,Z,volumeNode):
# https://slicer.readthedocs.io/en/latest/developer_guide/script_repository/volumes.html "Get volume voxel coordinates from markup control point RAS coordinates"
    point_Ras = [X,Y,Z]

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

###############################################################################    
###############################################################################    
###############################################################################    

def getIJKCoordinates2(roiNode,volumeNode):
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

###############################################################################    
###############################################################################    
###############################################################################    

def affine_registration(fixed,moving):
    
    R = sitk.ImageRegistrationMethod()

    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,
                                            minStep=1e-12,
                                            numberOfIterations=1000,
                                            gradientMagnitudeTolerance=1e-8)
    R.SetOptimizerScalesFromIndexShift()
    tx = sitk.CenteredTransformInitializer(fixed, moving,
                                        sitk.Similarity2DTransform())
    R.SetInitialTransform(tx)
    R.SetInterpolator(sitk.sitkLinear)

    outTx = R.Execute(fixed, moving)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)

    return out


def gamma_evaluation(ev,ref,ROW_START,ROW_END,COL_START,COL_END,spacing,doseTol,distTol,doseThreshold):
    
    BOX = math.ceil(distTol/spacing)
    deltaDose = np.max(ev)*doseTol/100
    minDose = np.max(ev)*doseThreshold/100

    distBox = np.zeros((2*BOX+1,2*BOX+1),dtype=np.float32)
    for i in range(distBox.shape[0]):
        for j in range(distBox.shape[1]):
            distBox[i,j] = (i-BOX)**2 + (j-BOX)**2

    distBox = distBox*(spacing**2/distTol**2)

    #CImg(distBox).display('distBox');

    gamma = np.zeros(ref.shape,dtype=np.float32)

    #workDir = os.path.dirname(__file__)
    #f = open(workDir + '/log.txt','a')
    #print(ev.shape,ref.shape,ROW_START,ROW_END,COL_START,COL_END,spacing,doseTol,distTol,doseThreshold,BOX,deltaDose,minDose,np.max(ev),file = f)

    for i in range(ref.shape[0]):
        for j in range(ref.shape[1]):

            if ev[ROW_START + i,COL_START + j] < minDose:
                continue

            box_ev = ev[ROW_START+i-BOX:ROW_START+i+BOX+1,COL_START+j-BOX:COL_START+j+BOX+1]
            box_ref = np.ones(box_ev.shape,dtype=np.float32)*ref[i,j]
            val = np.min( (box_ev - box_ref)**2/deltaDose**2 + distBox)
            #print(i,j,val,file=f)
            gamma[i,j] = val

    #f.close()

    return gamma
  

class QA(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("QA")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#QA">module documentation</a>.
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
# QAParameterNode
#


@parameterNodeWrapper
class QAParameterNode:
    """
    The parameters needed by module.
    """

    filmVolume: vtkMRMLScalarVolumeNode
    tpsDoseVolume: vtkMRMLScalarVolumeNode
    roi: vtkMRMLMarkupsROINode

#
# QAWidget
#


class QAWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/QA.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = QALogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        self.ui.labelX.setVisible(False)
        self.ui.labelY.setVisible(False)
        self.ui.labelZ.setVisible(False)
        self.ui.doubleSpinBoxX.setVisible(False)
        self.ui.doubleSpinBoxY.setVisible(False)
        self.ui.doubleSpinBoxZ.setVisible(False)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

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
        if not self._parameterNode.filmVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.filmVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[QAParameterNode]) -> None:
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
        if self._parameterNode and self._parameterNode.filmVolume and self._parameterNode.tpsDoseVolume:
            self.ui.applyButton.toolTip = _("Compute gamma index")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select reference and evaluated volume nodes")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""

#        workDir = os.path.dirname(__file__)
#        f = open(workDir + '/log.txt','w')
#        print(self._parameterNode.tpsDoseVolume.GetStorageNode().GetFileName(),file = f)
#        f.close()
#
        
        TPSDoseFileName = self._parameterNode.tpsDoseVolume.GetStorageNode().GetFileName()
        ds = pydicom.dcmread(TPSDoseFileName)
        scaling = float(ds.DoseGridScaling)

        dose = slicer.util.arrayFromVolume(self._parameterNode.tpsDoseVolume)*scaling
        film = slicer.util.arrayFromVolume(self._parameterNode.filmVolume)[0]/100

        path = self.ui.RTPlanPath.currentPath
        ds = pydicom.dcmread(path)
        X,Y,Z = ds.BeamSequence[0].ControlPointSequence[0].IsocenterPosition

        I, J, K = getIJKCoordinates1(X,-Y,Z,self._parameterNode.tpsDoseVolume)
        #I, J, K = getIJKCoordinates1(self.ui.doubleSpinBoxX.value,self.ui.doubleSpinBoxZ.value,self.ui.doubleSpinBoxY.value,self._parameterNode.tpsDoseVolume)

        ROW_START = 0
        ROW_END = film.shape[0]
        COL_START = 0
        COL_END = film.shape[1]
        spacing = self._parameterNode.filmVolume.GetSpacing()

        if self._parameterNode.roi is not None:
            col,row,_ = getIJKCoordinates2(self._parameterNode.roi,self._parameterNode.filmVolume)
            size = self._parameterNode.roi.GetSize()
            colSpan,rowSpan,_ = [int(s/sp) for s,sp in zip(size,spacing)]
            ROW_START = int(row - rowSpan//2)
            ROW_END = int(ROW_START + rowSpan)
            COL_START = int(col - colSpan//2)
            COL_END = int(COL_START + colSpan)

        cropped = np.copy(film[ROW_START:ROW_END,COL_START:COL_END])

        section = np.copy(dose[:,J,:])
        section = section[::-1,:]

        moving = sitk.GetImageFromArray(section)
        fixed = sitk.GetImageFromArray(cropped)

        out = affine_registration(fixed,moving)
        out = sitk.GetArrayFromImage(out)


        gamma_map = gamma_evaluation(film,out,ROW_START,ROW_END,COL_START,COL_END,spacing[0],self.ui.spinBoxPercentDose.value,self.ui.doubleSpinBoxDTA.value,self.ui.spinBoxDoseThreshold.value)

        GPR = (1.0 - len(np.where(gamma_map>=1.0)[0])/np.prod(gamma_map.shape))*100

        self.ui.labelGPR.text = f'Gamma index = {GPR:.2f}'
     
        #CImg(section).display('section');
        CImg(cropped).display('film');
        CImg(out).display('registered TPS dose');
        CImg(gamma_map).display('gamma image');

        nodeName = 'cropped film'
        crossSectionNode1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", nodeName)
        slicer.util.updateVolumeFromArray(crossSectionNode1, cropped)    

        nodeName = 'registered TPS dose'
        crossSectionNode1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", nodeName)
        slicer.util.updateVolumeFromArray(crossSectionNode1, out)    

        nodeName = 'gamma image'
        crossSectionNode1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", nodeName)
        slicer.util.updateVolumeFromArray(crossSectionNode1, gamma_map)    
        

#
# QALogic
#


class QALogic(ScriptedLoadableModuleLogic):
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
        return QAParameterNode(super().getParameterNode())

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
# QATest
#


class QATest(ScriptedLoadableModuleTest):
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
        self.test_QA1()

    def test_QA1(self):
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
        inputVolume = SampleData.downloadSample("QA1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = QALogic()

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
