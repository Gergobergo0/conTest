## Holographic Microscope Focus Estimation – Neural Network Challenge

As part of a competition supported by *HUN-REN SZTAKI*, our goal was to develop a deep learning system capable of estimating the absolute focus distance of cells from images captured by a holographic microscope.  
The task was a **regression problem**, evaluated using **root mean squared error (RMSE)**.  
The final model was based on a fine-tuned ResNet, using merged amplitude and phase images as input. We implemented a custom learning rate scheduler and achieved a score that tied us for 1st place.

📄 **Documentation:** [Link](https://github.com/Gergobergo0/conTest/blob/main/DOCUMENTATION.pdf)

![Focus Estimation Result](https://github.com/user-attachments/assets/ddab4810-7a5a-405a-b382-d0f825052909)
