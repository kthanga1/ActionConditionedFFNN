# ************** STUDENTS EDIT THIS FILE **************

from SteeringBehaviors import Wander
import SimulationEnvironment as sim

import numpy as np

# from algo.readdata import readfile


def collect_training_data(total_actions):
    #set-up environment
    sim_env = sim.SimulationEnvironment()

    #robot control
    action_repeat = 100
    steering_behavior = Wander(action_repeat)

    num_params = 7
    #STUDENTS: network_params will be used to store your training data
    # a single sample will be comprised of: sensor_readings, action, collision
    network_params = []

    for action_i in range(total_actions):
        progress = 100*float(action_i)/total_actions
        print(f'Collecting Training Data {progress}%   ', end="\r", flush=True)

        #steering_force is used for robot control only
        action, steering_force = steering_behavior.get_action(action_i, sim_env.robot.body.angle)
        sensor_readings = []
        collision = 0
        for action_timestep in range(action_repeat):
            if action_timestep == 0:
                _, collision, sensor_readings = sim_env.step(steering_force)
            else:
                _, collision, sensor_readings = sim_env.step(steering_force)

            if collision:
                steering_behavior.reset_action()
                #STUDENTS NOTE: this statement only EDITS collision of PREVIOUS action
                #if current action is very new.
                if action_timestep < action_repeat * .3: #in case prior action caused collision
                    network_params[-1][-1] = collision #share collision result with prior action
                break
        network_params.append([sensor_readings[0],sensor_readings[1],sensor_readings[2],sensor_readings[3], sensor_readings[4], action, collision])
        #STUDENTS: Update network_params.   x
    network_params = np.array(
              network_params)
    np.savetxt('submission.csv', network_params, delimiter=',', fmt='%f')


    #STUDENTS: Save .csv here. Remember rows are individual samples, the first 5
    #columns are sensor values, the 6th is the action, and the 7th is collision.
    #Do not title the columns. Your .csv should look like the provided sample.



if __name__ == '__main__':
    total_actions = 1000
    collect_training_data(total_actions)
