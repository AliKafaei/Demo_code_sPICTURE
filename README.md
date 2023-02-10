# Demo_code_sPICTURE
Demo code for Lateral Strain Imaging using Self-supervised and Physically Inspired Constraints in Unsupervised Regularized Elastography accepted in IEEE Transaction in Medical Imaging [1]. The network weights trained on experimental phantom data is provided here.  

## Install
The network architecture is MPWC-Net++ [2] which is a modified variant of PWC-Net irr [3]. Please follow the installation guide of original implementation of PWC-Net irr. 

## Results 
You should get the followng results from the demo code.
<div style="width: 40%; height: 40%">
  
 ![](https://github.com/AliKafaei/Demo_code_sPICTURE/blob/main/Axial_Strain.PNG)
  ![](https://github.com/AliKafaei/Demo_code_sPICTURE/blob/main/Lateral_Strain.PNG)

## Updates (2023-02-10)
Networks weights trained on simulation data and using Local Normalized Cross Correlation (LNCC) as the data loss is added (file name: sPICTURE_Simulation.pth.tar).  
</div>
If you use this code, please cite [1],[2], and [3].

## References
### [1] 
@article{tehrani2022lateral,
  title={Lateral Strain Imaging using Self-supervised and Physically Inspired Constraints in Unsupervised Regularized Elastography},
  author={Tehrani, Ali KZ and Ashikuzzaman, Md and Rivaz, Hassan},
  journal={IEEE Transactions on Medical Imaging},
  year={2022},
  publisher={IEEE}
}
### [2] 
@inproceedings{tehrani2021mpwc,
  title={MPWC-Net++: evolution of optical flow pyramidal convolutional neural network for ultrasound elastography},
  author={Tehrani, Ali KZ and Rivaz, Hassan},
  booktitle={Medical Imaging 2021: Ultrasonic Imaging and Tomography},
  volume={11602},
  pages={14--23},
  year={2021},
  organization={SPIE}
}
### [3]
@inproceedings{hur2019iterative,
  title={Iterative residual refinement for joint optical flow and occlusion estimation},
  author={Hur, Junhwa and Roth, Stefan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5754--5763},
  year={2019}
}
