import numpy as np
from maze import Maze, Particle, Robot
import bisect
import rospy
from gazebo_msgs.msg import  ModelState
from gazebo_msgs.srv import GetModelState
import shutil
from std_msgs.msg import Float32MultiArray
from scipy.integrate import ode

import random

def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return [roll, pitch, yaw]

def vehicle_dynamics(t, vars, vr, delta):
    curr_x = vars[0]
    curr_y = vars[1] 
    curr_theta = vars[2]

    dx = vr * np.cos(curr_theta)
    dy = vr * np.sin(curr_theta)
    dtheta = delta
    return [dx,dy,dtheta]

class particleFilter:
    def __init__(self, bob, world, num_particles, sensor_limit, x_start, y_start):
        self.num_particles = num_particles  # The number of particles for the particle filter
        self.sensor_limit = sensor_limit    # The sensor limit of the sensor
        particles = list()

        ##### TODO:  #####
        # Modify the initial particle distribution to be within the top-right quadrant of the world, and compare the performance with the whole map distribution.
        for i in range(num_particles):

            # (Default) The whole map
            #x = np.random.uniform(0, world.width)
            #y = np.random.uniform(0, world.height)


            ## first quadrant
            x = np.random.uniform(world.width/2, world.width)
            y = np.random.uniform(world.height/2, world.height)

            particles.append(Particle(x = x, y = y, maze = world, sensor_limit = sensor_limit))

        ###############

        self.particles = particles          # Randomly assign particles at the begining
        self.bob = bob                      # The estimated robot state
        self.world = world                  # The map of the maze
        self.x_start = x_start              # The starting position of the map in the gazebo simulator
        self.y_start = y_start              # The starting position of the map in the gazebo simulator
        self.modelStatePub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.controlSub = rospy.Subscriber("/gem/control", Float32MultiArray, self.__controlHandler, queue_size = 1)
        self.control = []                   # A list of control signal from the vehicle
        return

    def __controlHandler(self,data):
        """
        Description:
            Subscriber callback for /gem/control. Store control input from gem controller to be used in particleMotionModel.
        """
        tmp = list(data.data)
        self.control.append(tmp)

    def getModelState(self):
        """
        Description:
            Requests the current state of the polaris model when called
        Returns:
            modelState: contains the current model state of the polaris vehicle in gazebo
        """

        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            modelState = serviceResponse(model_name='polaris')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
        return modelState

    def weight_gaussian_kernel(self,x1, x2, std = 5000):
        if x1 is None: # If the robot recieved no sensor measurement, the weights are in uniform distribution.
            return 1./len(self.particles)
        else:
            tmp1 = np.array(x1)
            tmp2 = np.array(x2)
            return np.sum(np.exp(-((tmp2-tmp1) ** 2) / (2 * std)))


    def updateWeight(self, readings_robot):
        """
        Description:
            Update the weight of each particles according to the sensor reading from the robot 
        Input:
            readings_robot: List, contains the distance between robot and wall in [front, right, rear, left] direction.
        """

        ## TODO #####
        weights = []
        for i in range(self.num_particles):
            readings_particle = self.particles[i].read_sensor()
            weights.append(self.weight_gaussian_kernel(readings_robot, readings_particle))

        # Normalize the weights
        norm = np.linalg.norm(weights)
        norm_weights = weights / norm
        for i in range(self.num_particles):
            self.particles[i].weight = norm_weights[i]
        
        ###############
        # pass

    def resampleParticle(self):  #########      wrong
        """
        Description:
            Perform resample to get a new list of particles 
        """
        particles_new = list()

        ## TODO #####

        weights = []
        for i in range(self.num_particles):
            weights.append(self.particles[i].weight)

        # rnd = np.random.uniform(0,1)
        # index = int(rnd * (self.num_particles - 1))
        # beta = 0.0
        # max_weight = max(weights)
        # for particle in self.particles:
        #     beta += np.random.uniform(0,1) * 2.0 * max_weight
        #     while beta > weights[index]:               # weights[index] = random weight
        #         beta -= weights[index]
        #         index = (index + 1) % self.num_particles

        #     particle = self.particles[index]
        #     particles_new.append(Particle(x = particle.x, y = particle.y, heading = particle.heading, maze = particle.maze, sensor_limit = particle.sensor_limit))

        cumsum = np.cumsum(weights)
        for i in range(self.num_particles):
            rnd = np.random.uniform(0,cumsum[-1])    # random index = np.random.randint(0,cumsum[-1])
            index = 0
            for w in cumsum:
                if w > rnd:
                    break
                index += 1
            particle = self.particles[index]
            particles_new.append(Particle(x = particle.x, y = particle.y, heading = particle.heading, maze = particle.maze, sensor_limit = particle.sensor_limit))
        ###############

        self.particles = particles_new

    def particleMotionModel(self):    ######### wrong
        """
        Description:
            Estimate the next state for each particle according to the control input from actual robot 
        """
        ## TODO #####

        if(len(self.control) == 0):
            return

        # vr = np.array([c[0] for c in self.control])
        # delta = np.array([c[1] for c in self.control])
        vr = self.control[-1][0]
        delta = self.control[-1][1]
       
        for i in range(self.num_particles):
            initR = [self.particles[i].x, self.particles[i].y, self.particles[i].heading]    
            r = ode(vehicle_dynamics)
            r.set_initial_value(initR)
            r.set_f_params(vr, delta)
            val = r.integrate(r.t + 0.01)

            self.particles[i].x = val[0]
            self.particles[i].y = val[1]
            self.particles[i].heading = val[2]

        ###############
        # pass


    def runFilter(self):
        """
        Description:
            Run PF localization
        """
        count = 0 
        while True:
            ## TODO: (i) Implement Section 3.2.2. (ii) Display robot and particles on map. (iii) Compute and save position/heading error to plot. #####
            self.particleMotionModel()
            reading = self.bob.read_sensor()
            self.updateWeight(reading)
            self.resampleParticle()

            self.world.show_particles(self.particles)
            self.world.show_estimated_location(self.particles)
            self.world.show_robot(self.bob)
            self.world.clear_objects()
            ###############
