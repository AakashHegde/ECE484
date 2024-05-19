# Obstacle Sensing Autonomous Car

For a more detailed walkthrough, view the [project PDF](https://github.com/AakashHegde/ECE484/blob/main/ECE484_final_presentation.pdf).

### Motivation
- **Enhancing Road Safety**: Ensuring lane discipline, maintaining safe speeds, and following traffic rules by recognizing traffic signs.
- **Ensuring Pedestrian Safety**: Addressing rising pedestrian fatalities.
- **Improving Transportation Efficiency**: Reducing traffic congestion through orderly traffic management.

### System Overview
**Primary Goals**:
- Drive autonomously within the GEM Car track using only a camera for perception.
- Stay within designated track lanes at all times.
- Ensure passenger comfort during turns and braking.

**Additional Capabilities**:
- Detect and stop for pedestrians.
- Detect and stop at stop signs.

### Perception and Detection
- **Lanes**: Utilize camera input and image processing to detect lane lines and set waypoints.
- **Obstacles**: Use YOLOv8m, a pre-trained CNN-based model, to detect pedestrians, stop signs, and other vehicles in real-time.

### Localization and Waypoints
- Localize using only camera input (no GPS or IMU).
- Generate waypoints based on lane line visibility and adjust according to the car's speed.

### Control
- Proximity determined by bounding box area around detected objects. Follow waypoints using a Pure Pursuit Controller.
- **Additional Considerations**: 
    - Synthesize waypoints when lanes are not detected. 
    - Ensure passenger comfort by slowing down during turns and braking gradually.

### Results
- **Simulation**: Utilized Gazebo simulator to refine lane detection, waypoint algorithms, and control strategy.
- **Real-World Tests**: Demonstrated lane following, stop sign detection, and pedestrian detection in various scenarios.

### Challenges and Limitations
- **Lane Detection**: Issues with lane stabilization, narrow lanes, and limited camera field of view.
- **Car Dynamics Code**: Complexity in understanding PID for steering and speed.
- **Weather Conditions**: Performance affected by weather (cloudy, sunny, rain, snow).

### Future Work
- Improve lane detection consistency and waypoint following for smoother steering and better lane maintenance at higher speeds.


### Branches
**main:** primarily MP submissions
**gem_branch:** code used on the GEM car
**sim:** code used for Highbay simulation on Gazebo