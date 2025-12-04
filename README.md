# RadarStream

RadarStream is a real-time RAWDATA acquisition, processing, and visualization system for TI MIMO mmWave radar series.



https://github.com/user-attachments/assets/7ce99b51-a1af-4025-8a84-ee580eb92d04

Demo1: Real-time Motion Detection and Radar Feature Visualization
<figure>
  <img src="img/4.gif" alt="图片描述" width="100%">
  <figcaption>Demo2: Real-time Gesture Recognition System</figcaption>
</figure>

## Project Overview

This system supports Texas Instruments' MIMO mmWave radar series for real-time raw data acquisition, processing, and visualization. In addition to the RF evaluation board, the DCA1000EVM is required for data capture. Currently, the system has been tested with:
- IWR6843ISK
- IWR6843ISK-OBS
- IWR1843ISK

If you encounter any issues while using this project, please feel free to submit a pull request.

## Features ✨

*   **Real-time, Multi-threaded Radar Data Acquisition from TI MIMO mmWave Radar Sensors:**
    *   Leveraging a **multi-threaded architecture 🧵** for data acquisition and processing.
    *   To overcome Python's Global Interpreter Lock (GIL) and enable true multi-core processing, the data acquisition module is **wrapped in C 🚀**, ensuring near real-time, frame-loss-free data capture and handling.
*   **Multi-dimensional Feature Extraction:**
    *   Range-Time Information (RTI) 
    *   Doppler-Time Information (DTI) 
    *   Range-Doppler Information (RDI) 
    *   Range-Azimuth Information (RAI) 
    *   Range-Elevation Information (REI) 
*   **Interactive Visualization Interface**

## Requirements

- Python 3.6+
- PyQt5
- PyQtGraph
- NumPy
- PyTorch
- Matplotlib
- Serial

## Hardware Requirements

- TI MIMO mmWave Radar Sensor (tested with IWR6843ISK and IWR6843ISK-OBS)
- DCA1000 EVM (essential for raw data capture)
- PC with Windows OS 

## Firmware Requirements
The firmware must be selected from the `mmwave_industrial_toolbox_4_10_1\labs\Out_Of_Box_Demo\prebuilt_binaries/` directory inside any version of the mmwave_industrial_toolbox.  
There is no strict requirement to use version 4.10.1.

## Setup and Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install pyqt5 pyqtgraph numpy torch matplotlib pyserial
   ```
3. Connect the mmWave radar sensor and DCA1000 EVM to your computer (only need a 5V 3A DC power wire,  a Ethernet Cable, and a micro USB wire)
4. Configure the network IPv4 settings (referencing the IPv4 configuration process from using mmWaveStudio for the DCA1000 EVM)

Two different acquisition methods are shown here: one figure displays Raspberry Pi 4B acquisition, while the other demonstrates Windows-based  acquisition. However, the Raspberry Pi acquisition has very few frames during real-time processing and display, making it prone to data loss. (not recommended to use Raspberry Pi for acquisition)

<p align="center">
  <img src="img/1.png" width="36%" />
  <img src="img/2.png" width="45%" />
  <img src="img/6.jpg" width="40.5%" />
  <img src="img/7.jpg" width="40.5%" />
</p>

## 3D Printed Mount

The repository includes STL files for a 3D printed structure designed to mount and secure the DCA1000EVM board.

**Note:** You will need some M3 size nylon standoffs and screws for assembly.

<p align="center">
  <img src="img/3.png" width="70%" />
</p>

## Usage

1. Run the main application:
   ```
   python main.py
   ```
2. Select the appropriate COM port for the radar CLI interface
3. Choose a radar configuration file
4. Click "Send Config" to initialize the radar
5. Use the interface to:
   - Visualize radar data in real-time
   - Capture training data for machine learning models


## Project Structure
- `config/`: Configuration files for different radar settings
- `gesture_icons/`: Gesture icons for visualization
- `libs/`: Library files for radar communication
- `STL3D`: 3D printed mount STL files
- `main.py`: Main application entry point
- `real_time_process.py`: Real-time data processing
- `radar_config.py`: Radar configuration utilities
- `iwr6843_tlv/`: TLV protocol implementation for IWR6843
- `dsp/`: Digital signal processing modules
- `UI_interface.py`: PyQt5 user interface


## Citation

If this project helps your research, please consider citing our papers that are closely related to this tool:

```
@ARTICLE{11270504,
  author={Chen, Qin and Lu, Qunfeng and Chen, Yaoxi and Tian, Yu and Cui, Zongyong and Cao, Zongjie},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Domain-Generalized Gesture Recognition via mmWave Radar Signal Multi-View Learning}, 
  year={2025},
  doi={10.1109/TIM.2025.3637962}}

@ARTICLE{10714388,
  author={Chen, Qin and Cui, Zongyong and Tian, Yu and Chen, Yaoxi and Cao, Zongjie},
  journal={IEEE Internet of Things Journal}, 
  title={Joint Position Estimation for Hand Motion Using MIMO FMCW mmWave Radar}, 
  year={2025},
  volume={12},
  number={3},
  pages={2838-2853},
  doi={10.1109/JIOT.2024.3478234}}

@ARTICLE{10288185,
  author={Chen, Qin and Cui, Zongyong and Zhou, Zheng and Tian, Yu and Cao, Zongjie},
  journal={IEEE Internet of Things Journal}, 
  title={MMHTSR: In-Air Handwriting Trajectory Sensing and Reconstruction Based on mmWave Radar}, 
  year={2024},
  volume={11},
  number={6},
  pages={10069-10083},
  doi={10.1109/JIOT.2023.3325258}}

```


## Acknowledgements

This project references and builds upon:
- [real-time-radar](https://github.com/AndyYu0010/real-time-radar) by AndyYu0010
- [OpenRadar](https://github.com/PreSenseRadar/OpenRadar) - specifically the DSP module

## TODO

Future improvements planned for this project:
- [ ] Validate compatibility with more RF evaluation boards
- [ ] Migrate from PyQt5 to PySide6
- [ ] Make the API in the libs folder more flexible
