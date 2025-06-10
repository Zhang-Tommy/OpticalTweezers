# SMARTS Lab Holographic Optical Tweezers Documentation
Tommy Zhang  
June 9 2025

## Preface
The work I did for my research under Professor Ashis Banerjee mostly involved developing/finding new strategies and implementing them into the holographic optical tweezer (HOT) system we have.  
I will assume the reader has a basic understanding of the principles of HOTs and have gone through the documentation provided with the Meadowlark Optics branded HOT system (Red Tweezers, User Guide, etc...).

## Code Organization
The code and supporting files are all contained within the *Code* directory. Under this directory are the two main projects I worked on for my research.

- *python_controller* contains the files required for running the MPC controller with the HOT system. It can interface with the camera and spatial light modulator (SLM) of the HOT system. There is also a GUI for the user to view the workspace camera, add traps, and start MPC control instances (which navigate traps from a start to goal location)
- *phase_predictor* contains the supporting files for generating, training, and validating a neural network for the task of phase retrieval. Phase retrieval is the process of generating the phase mask for display onto the SLM. Please refer to the MATLAB toolbox *otslm* for greater documentation on some implementations used here.

## Python Controller
### main.py
main.py is the python file to be executed for starting up the HOT controller. It contains the main calling logic for instantiating control instances, the hologram engine, and the camera. holo is the main control loop for a single guided trap. Using the multiprocessing package allows us to spawn multiple instances of the holo controller while sharing state information.  
The simulator function is used for bench testing/continuous integration of the HOT control system. Any changes made to the control strategy, bead detection, and other subsystems interacting with the tweezers can be validated using the simulator.  
The cam function is called rather than the simulator when used on the actual HOT system. It is analogous to the simulator function but uses the real camera feed for display onto the opencv window rather than the simulator view. The cam function is responsible for initializing the camera hardware, then managing user inputs to the opencv GUI. The controlled trap's goal positions and start positions are defined here and instances of the holo function are spawned here again using multiprocessing.
The clear_region and ctrl functions are used to automatically clear some desired region in the workspace of detected micro-beads. The motivation for this was for conducting experiments where micro-structure formations were desired and having many micro-beads in the center of the workspace meant no other room for assembly.

### camera.py
camera.py contains the main logic for image aquisition from the systems GigE high frame rate camera. It also contains the bead detection code which detects the centroid positions of all micro-beads in the workspace. These are referred to as keypoints in opencv and the rest of the control logic. Note that we cannot simply use the traditional image input device on the computer because the camera used works on the GigE standard (ethernet connection). This is why the harvester package is used in addition to opencv for acquiring images.

### mpc.py
mpc.py contains the entire implementation for the iLQR MPC online optimization control strategy used for guiding traps. This implementation was derived from an example from AA 548 Linear Multivariable Control Spring 2024. The main modifications are in the Dynamics, Running Cost, and Terminal Cost functions. Where we included our own dyanmics model for the optical tweezers system (spring mass damper), and custom cost functions for the dynamics used. In addition, the implementation was extended for use on live state updates (rather than being deterministic in the example) based on the camera bead detection. Also, additional logic was added to work with the three complex traps used in our experiments (annular, line, and point traps). The line traps include a custom obstacle boundary modeled using an ellipse. 

### spot_manager.py
The spot and spot_manager files are the main classes for the HOT control system. These classes allow for easily managing an arbitrary number of traps that can be added into the workspace. 

Spot manager keeps track of all optical traps active in the workspace along with the starting and goal positions of those traps. In addition, it contains data structures for storing the obstacle positions. The spot manager class is registered in multiprocessing such that the spot manager object state can be shared through the other running processes. This enables multiple control instances to run concurrently and know where other controlled traps are located.

### spot.py
The Spot class is the base class for keeping track of a single optical traps state. This includes the parameters sent to the hologram engine and parameters which are used to indicate the type and location of the optical trap. Spot implements functions which can be called to change specific attributes such as position, intenisty, and phase.

### utilities.py 
The utilities.py file contains low-level or lengthy control logic whose purpose can easily be inferred from the name. 

### constants.py
The constants.py file contains parameters which can be changed to tune the behavior of the tweezers control system, as well as adjust for a different physical system architecture (different camera, calibration coefficients).

This is also where the user can toggle on and off the debug and simulator modes. The debug mode draws extra information such as the DBSCAN clustering and keypoint dots onto the GUI. Whereas the simulator will enable the simulator implementation rather than the hardware camera acquisition.

## Phase Prediction
The U-Net project files are contained under the phase_prediction subdirectory in the phase_predictor branch. I will assume the reader has experience with pytorch and an understanding of the U-Net architecture and training/validation process.

### parallel_input_gen.py
This is the data generation script that will use a ported version of red tweezers in Python to generate batches of training data for training the U-Net. The training data are sets of inputs (intensity distributions) and outputs (phase masks) of the same dimensionality. 

The training data is generated by randomly pulling from uniform distributions which determine trap location, trap type, and number of traps. The choice of distribution is extremely important for training a successful U-Net, the current distribution of 80/20/20 point/annular/line traps worked well for training U-Net models of all dimensionality.

### input_generator.py
input_generator.py is the simplified version of parallel_input_gen.py which generates the input data sequentially through a loop rather than parallelizing the generation over multiple cpu cores.

### phase_calculator.py
This is the Red Tweezers gpu code ported into Python. It accepts a spot parameters array analagous to the Red Tweeer hologram engine program and outputs a phase mask for display onto the slm. 

The phase calculation is similarly sped up by parallelizing the computation using JAX. Though the Python version does have slight overhead and is not 1:1 in terms of performance with the hologram engine.

There are two functions which can be used: calculate_phase_mask and batch_calculate_masks. The first calculates a single phase mask given a single spot array, the batch calculator will output multiple phase masks given multiple spot arrays.

### unet.py
This contains the U-Net architecture implemented using PyTorch, as well as the dataloader for properly loading training samples from the hdf5 data file.

### train.py
train.py contains the training code for the U-Net. The model training hyperparameters can be tuned here.

## Misc Issues
Development for these two main projects were done on a personal workstation. The laptop which is used for operating the HOT system is quite old and still runs Windows 7. Upgrading the OS is really not an option due to the computers old hardware. This means that many workarounds were done to make more modern packages like opencv, JAX, and PyTorch run on the system. These package versions are mainly the older versions which are still compatible with Windows 7 and older versions of Python but **restarting development should be done using more modern hardware**. 

Physical misalignments and optical imperfections in the HOT system presents itself when maneuvering micro-beads in the workspace using the Meadowlark Optics provided labview application. The positioning of optical traps is not 1:1 with the desired position indicated by the user and the misalignment severity changes based on the location of the trap. In addition, the perceived power (strength of the optical trap) will change based on the distance away the trap is from the zero-order spot located at the top-left of the workspace. The camera is positioned such that the zero-th order spot is centered at top left corner. This zero-th order spot is simply where the laser is incident if there is nothing being addressed to the SLM. As a point trap is placed light is diffracted such that a high intensity diffraction limited point is incident on the sample plane. The sample plane is also considered the sample as the far-field when discussing fourier optics due to the high NA microscope objective being used.

## Quick Start Guide
After reading through the documentation for the HOT system provided by Meadowlark Optics this quick start guide can be used to get the system up and running again.

**Gather materials:**
- Microscope slide
- Microscope slide cover
- Diluted solution of micro-beads (refer to past paper for dilution ratio)
- 10uL pipette and pipette tip
- Vibratory shaker/mixer  

1. Power on the CUBE (switch at the back of the system)
2. Power on the stage control (switch at the back of stage controller)
3. Ensure all cables are properly plugged into the laptop (DisplayPort for SLM, ethernet for Camera, USB for stage control and LED control)
4. Launch LEDDriver.exe program and choose the LED camera port. Set the LED power to around 250mW
5. Launch the labview gui application for controlling the tweezers. Start the labview program within labview and verify the camera feed and LED is working (black screen if LED is off, gray camera feed if LED is on. Can also look at LED to verify)
6. Prepare the micro-bead sample by first using the vibratory mixer to thoroughly mix the micro-bead and DI water solution for around 20 seconds. Use the pipette to dispense around 20uL of the solution on the convex well of the microscope slide and place the slide cover over the solution. Ensure there are no significant air bubbles. 
7. Apply a single drop of objective lens immersion oil onto the objective
8. Place the microscope slide with the slide cover facing down onto the objective. Use the fixtures to hold the microscope slide in place and center the sample with the objective. 
9. Close the laser safety cover and view the sample with the camera. Change the z-height to focus the camera onto the micro-beads. Moving the stage in the xy direction can also help find micro-beads.
10. Turn the laser on to 35% power and use the labview GUI to trap/maneuver your first beads!
11. Afterwards, clean the microscope slides and wipe the immersion oil off the objective. Turn off the entire HOT system using the power switches at the back.

**Warnings**
- Never forget to turn off the laser before opening the laser cover. Use the red E-stop to ensure the laser is not running and cannot easily be turned back on accidentally. 
- There are safety switches which will automatically turn off the laser if the cover opens. This should not be depended on while operating the HOT system.
- Using the microscope slide is the best method for preparing the sample for the HOT system. Glass bottom petri dishes were used initially but the non-uniformity of the solution (due to surface tension creating a droplet) meant that micro-beads would congregate quickly at the zero-th order spot during operation of the laser. The microscope slide does not have this issue and is preferred. 
- Ensure the microscope slide and cover is clean. If there are noticeable streaks while looking through the camera it can inhibit proper operation of the tweezers. 
- 


