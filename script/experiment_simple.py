#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from typing import List

# Import ROS Messages
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose
from multimodal_fusion.msg import FusedCommand, TrajectoryError
from manipulator_planner.srv import PlanningParameters, PlanningParametersRequest, PlanningParametersResponse

# Move Robot Utilities
from utils.move_robot import UR10e_RTDE_Move, GRIPPER_OPEN, GRIPPER_CLOSE
from utils.command_list import *
from utils.object_list import HOME, SPECIAL_PLACE, WAIT_POSITION, object_list, area_list

"""

Labelling:

    0: Empty Command
    1: Place Object + Given Area (Voice Only)
    2: Place Object + Given Area + Point-At
    3: Place Object + Point-At
    4: Point-At (Gesture Only)
    5: Stop (Gesture Only)
    6: Pause (Gesture Only)

    7: Start Experiment
    8: Object Moved
    9: User Moved
    10: User Can't Move
    11: Replan Trajectory
    12: Wait for Command
    13: Can Go
    14: Wait Time

"""

class ExperimentManager(Node):

    # TTS Error Messages
    OBSTACLE_DETECTED_ERROR_STRING = "There is an obstacle where I have to place the object"
    MOVE_TO_USER_ERROR_STRING = "I have to get there, if you don't move I'll have to slow down too much"
    HELP_SPECIAL_BLOCK_ERROR_STRING = 'I need your help with this special block, tell me when you bring it and I release the gripper'
    ROBOT_STOPPED_SCALING_ERROR_STRING = 'I stopped due to safety limitations, please move away from the robot'

    # Flags
    experiment_started, stop = False, False
    wait_for_command, error_stop = False, False

    # Received Data
    error_handling_command, received_error = None, None

    def __init__(self) -> None:

        # Init ROS Node
        super().__init__('Experiment_Manager')
        self.node = rclpy.create_node('experiment_manager')

        # Instance Robot Movement Class
        self.robot = UR10e_RTDE_Move()

        # Publishers
        self.tts_pub = self.node.create_publisher(String, '/alexa/tts', 1)

        # Enable Obstacle Publisher
        self.enable_all_obstacles_pub = self.node.create_publisher(Bool, '/manipulator_planner/enable_all_obstacles', 1)
        self.enableGreenObstaclePub = self.node.create_publisher(Bool, '/manipulator_planner/enable_green_obstacle', 1)
        self.enableRedObstaclePub = self.node.create_publisher(Bool, '/manipulator_planner/enable_red_obstacle', 1)
        self.enableYellowObstaclePub = self.node.create_publisher(Bool, '/manipulator_planner/enable_yellow_obstacle', 1)
        self.enableSpecialObstaclePub = self.node.create_publisher(Bool, '/manipulator_planner/enable_special_obstacle', 1)

        # Stop Trajectory Publisher
        self.stop_trajectory_pub = self.node.create_publisher(Bool, '/trajectory_scaling/stop_trajectory', 1)

        # Subscribers
        self.command_sub = self.node.create_subscription(FusedCommand, '/multimodal_fusion/fused_command', self.command_callback, 1)
        self.trajectory_error_sub = self.node.create_subscription(TrajectoryError, '/trajectory_error', self.trajectory_error_callback, 1)


        # Set Parameters Service
        self.set_planning_parameters_service = self.node.create_client(PlanningParameters, '/manipulator_planner/set_planning_parameters')

        # Load Parameters
        self.gripper_enabled = self.node.get_parameter('/experiment/gripper_enabled').value

    def commandCallback(self, data:FusedCommand):

        self.node.get_logger().warn(f"Received Command: {data}")

        # Start / Stop Experiment Command -> Set Flags
        if   data.fused_command == START_EXPERIMENT: self.experiment_started = True; return
        elif data.fused_command in [STOP, PAUSE]: self.stop = True; return

        # Empty or Unused Commands -> Do Nothing
        elif data.fused_command in [EMPTY_COMMAND, POINT_AT]: return

        # CanGo Command -> Save CanGo Command
        elif data.fused_command in [CAN_GO] and self.wait_for_command: self.error_handling_command = data

        # Place Object Command -> Save Place Command
        elif data.fused_command in [PLACE_OBJECT_GIVEN_AREA, PLACE_OBJECT_GIVEN_AREA_POINT_AT, PLACE_OBJECT_POINT_AT]: self.error_handling_command = data

        # Voice-Only Commands -> Save Error Handling Command
        elif data.fused_command in [OBJECT_MOVED, USER_MOVED, WAIT_FOR_COMMAND, CAN_GO]: self.error_handling_command = data

    def trajectoryErrorCallback(self, data:TrajectoryError):

        # Save Received Error
        self.received_error = data.error
        self.node.get_logger().warn(f"Received Error: {data.error} | {data.info}")

    def handover(self, object_name, pick_position, place_position) -> bool:

        """ Handover Object """

        # Normal Velocity
        self.setPlanningParameters(velocity_factor=0.2)

        # Move Gripper to Starting Position | Negative Error Handling -> Return
        self.node.get_logger().info('Open Gripper')
        if not self.robot.move_gripper(GRIPPER_OPEN, self.gripper_enabled):
            if not self.errorHandling(OPEN_GRIPPER_ERROR, object_name): return False

        # Forward + Inverse Kinematic -> Increase z + 10cm
        self.node.get_logger().info('Forward Kinematic + Inverse Kinematic -> Increase z + 10cm')
        pick_position_cartesian: Pose = self.robot.FK(pick_position)
        pick_position_cartesian.position.z += 0.10
        pick_position_up: List[float] = self.robot.IK(pick_position_cartesian, pick_position)

        # Move 10cm Over the Object | Negative Error Handling -> Return
        self.node.get_logger().info('Move Over the Object')
        if not self.robot.move_joint(pick_position_up):
            if not self.errorHandling(MOVE_OVER_OBJECT_ERROR, object_name, pick_position_up): return False

        # Move to Object | Negative Error Handling -> Return
        self.node.get_logger().info('Move To the Object')
        if not self.robot.move_joint(pick_position):
            if not self.errorHandling(MOVE_TO_OBJECT_ERROR, object_name, pick_position): return False

        # Grip Object | Negative Error Handling -> Return
        self.node.get_logger().info('Close Gripper')
        if not self.robot.move_gripper(GRIPPER_CLOSE, self.gripper_enabled):
            if not self.errorHandling(CLOSE_GRIPPER_ERROR, object_name): return False
        rclpy.sleep(1)

        # Move 10cm Over the Object | Negative Error Handling -> Return
        self.node.get_logger().info('Move Over the Object')
        if not self.robot.move_joint(pick_position_up):
            if not self.errorHandling(MOVE_OVER_OBJECT_AFTER_ERROR, object_name, pick_position_up): return False

        # Forward + Inverse Kinematic -> Increase z + 10cm
        self.node.get_logger().info('Forward Kinematic + Inverse Kinematic -> Increase z + 10cm')
        place_position_cartesian: Pose() = self.robot.FK(place_position)
        place_position_cartesian.position.z += 0.10
        place_position_up: List[float] = self.robot.IK(place_position_cartesian, place_position)

        # Move 10cm Over the Place Position | Negative Error Handling -> Return
        self.node.get_logger().info('Move Over the Place Position')
        if object_name != 'special_block' and not self.robot.move_joint(place_position_up):
            if not self.errorHandling(MOVE_OVER_PLACE_ERROR, object_name, place_position_up): return False

        if object_name == 'special_block':

            self.setPlanningParameters(velocity_factor=0.1)
            rclpy.sleep(5)

        # Move to Place Position | Negative Error Handling -> Return
        self.node.get_logger().info('Move To the Place Position')
        if not self.robot.move_joint(place_position):
            if not self.errorHandling(MOVE_TO_PLACE_ERROR, object_name, place_position): return False

        if object_name == 'special_block':

            # Collaborative Release Object Routine
            self.collaborativeReleaseRoutine()
            return True

        # Release Object | Negative Error Handling -> Return
        self.node.get_logger().info('Open Gripper')
        if not self.robot.move_gripper(GRIPPER_OPEN, self.gripper_enabled):
            if not self.errorHandling(OPEN_GRIPPER_AFTER_ERROR, object_name): return False
        rclpy.sleep(1)

        # Move 10cm Over the Place Position | Negative Error Handling -> Return
        self.node.get_logger().info('Move Over the Place Position')
        if not self.robot.move_joint(place_position_up):
            if not self.errorHandling(MOVE_OVER_PLACE_AFTER_ERROR, object_name, place_position_up): return False

        # Normal Velocity
        self.setPlanningParameters(velocity_factor=0.2)

        # Move to Home | Negative Error Handling -> Return
        self.node.get_logger().info('Move To Home')
        if not self.robot.move_joint(HOME):
            if not self.errorHandling(MOVE_HOME_ERROR, object_name, HOME): return False

        return True

    def placeObject(self, place_area, object_name) -> bool:

        """ Place Object in Area """

        # FIX: Empty Area = 'front'
        if place_area == '': place_area = 'front'

        # Get Area Place Position
        for area in area_list:

            if area.name == place_area:

                # Desired Area Position Found
                place_position = area.position
                self.node.get_logger().info(f'Place Area Found: {place_area}')
                area_found = True
                break

            else: area_found = False

        # Area Not Found -> Return
            self.node.get_logger().error(f'Place Object Function | Place Area Not Found: {place_area} | Stop Experiment')
        else: self.node.get_logger().info(f'Place Object Function | Place Area Found: {place_area}')


        if object_name == 'special_block':

            # Forward + Inverse Kinematic -> Increase Place z + 5cm
            self.node.get_logger().info('Forward Kinematic + Inverse Kinematic -> Increase Place z + 5cm')
            place_position_cartesian: Pose() = self.robot.FK(place_position)
            place_position_cartesian.position.z += 0.05
            place_position: List[float] = self.robot.IK(place_position_cartesian, SPECIAL_PLACE)

        # Forward + Inverse Kinematic -> Increase z + 10cm
        self.node.get_logger().info('Forward Kinematic + Inverse Kinematic -> Increase z + 10cm')
        place_position_cartesian: Pose() = self.robot.FK(place_position)
        place_position_cartesian.position.z += 0.10
        place_position_up: List[float] = self.robot.IK(place_position_cartesian, place_position)

        # Move 10cm Over the Place Position | Error -> Return
        self.node.get_logger().info('Move Over the Place Position')
        if not self.robot.move_joint(place_position_up): return False

        # Move to Place Position | Error -> Return
        self.node.get_logger().info('Move To the Place Position')
        if not self.robot.move_joint(place_position): return False

        # Release Object | Error -> Return
        self.node.get_logger().info('Open Gripper')
        if not self.robot.move_gripper(GRIPPER_OPEN, self.gripper_enabled): return False
        rclpy.sleep(1)

        # Move 10cm Over the Place Position | Error -> Return
        self.node.get_logger().info('Move Over the Place Position')
        if not self.robot.move_joint(place_position_up): return False

        # Move to Home | Error -> Return
        self.node.get_logger().info('Move To Home')
        if not self.robot.move_joint(HOME): return False

        return True

    def errorHandling(self, handover_error, object_name, goal_position=None) -> bool:

        """ Error Handling """

        # Gripper Movement Error -> Stop Handover
        if handover_error in [OPEN_GRIPPER_ERROR, CLOSE_GRIPPER_ERROR, OPEN_GRIPPER_AFTER_ERROR]:

            self.node.get_logger().error('ERROR: An exception occurred during Gripper Movement -> Stop Experiment')
            self.error_stop = True
            return False

        # Pick Object Movement Error -> Stop Handover
        elif handover_error in [MOVE_OVER_OBJECT_ERROR, MOVE_TO_OBJECT_ERROR, MOVE_OVER_OBJECT_AFTER_ERROR]:

            self.node.get_logger().error('ERROR: An exception occurred during Pick Object Movement -> Stop Experiment')
            self.error_stop = True
            return False

        # Move Home Error -> Stop Handover
        elif handover_error in [MOVE_HOME_ERROR]:

            self.node.get_logger().error('ERROR: An exception occurred during Move to Home -> Stop Experiment')
            self.error_stop = True
            return False

        # Place Object Movement Error -> Check Error Type
        elif handover_error in [MOVE_OVER_PLACE_ERROR, MOVE_TO_PLACE_ERROR, MOVE_OVER_PLACE_AFTER_ERROR]:

            # Wait for Error Handling Command
            while self.received_error is None and not rclpy.ok(): self.node.get_logger().info_throttle(5, 'Waiting for Error Message')


            # Obstacle Detected -> Move to User
            if self.received_error == OBSTACLE_DETECTED_ERROR:

                # Publish Error Message
                error_msg = String()
                error_msg.data = self.OBSTACLE_DETECTED_ERROR_STRING
                self.ttsPub.publish(error_msg)
                self.node.get_logger().warn('ERROR: Obstacle Detected')

                # Wait for Error Handling Command
                while self.error_handling_command is None and not rclpy.ok():
                    self.node.get_logger().info_throttle(5, 'Waiting for Error Handling Command')

                # Object Moved -> Move to Place -> Restart Handover
                if self.error_handling_command.fused_command == OBJECT_MOVED:

                    self.node.get_logger().info('Object Moved Command -> Retry Place Object')

                    # Move to Position -> Positive Error Handling -> Continue Handover
                    if self.robot.move_joint(goal_position): return_flag = True

                    else:

                        # Negative Error Handling -> Stop Handover
                        self.node.get_logger().error('ERROR: An exception occurred during Object Moved Error Handling')
                        self.error_stop = True
                        return False

                # Put Object in Place Area -> Place Object -> Stop Handover
                elif self.error_handling_command.fused_command in [PLACE_OBJECT_GIVEN_AREA, PLACE_OBJECT_GIVEN_AREA_POINT_AT]:

                    # Place Object in Area Error Handling -> Stop Handover
                    if not self.placeObject(place_area=self.error_handling_command.area, object_name=object_name):

                        # Negative Error Handling -> Stop Handover
                        self.node.get_logger().error('ERROR: An exception occurred during Object Moved Error Handling -> Stop Experiment')
                        self.error_stop = True

                    # Stop Handover
                    return_flag = False

            # Move to User Error -> Stop Handover
            elif self.received_error == MOVE_TO_USER_ERROR:

                # Publish Error Message
                error_msg = String()
                error_msg.data = self.MOVE_TO_USER_ERROR_STRING
                self.ttsPub.publish(error_msg)
                self.node.get_logger().warn('ERROR: Move to User Error')

                # Move to Wait Position
                self.stopTrajectoryPub.publish(Bool())
                self.robot.move_joint(WAIT_POSITION, forced=True)

                # Wait for Error Handling Command
                while self.error_handling_command is None and not rclpy.ok():
                    self.node.get_logger().info_throttle(5, 'Waiting for Error Handling Command')

                # User Moved -> Move to Place -> Restart Handover
                if self.error_handling_command.fused_command in [USER_MOVED, CAN_GO]:

                    self.node.get_logger().info('User Moved Command -> Retry Place Object')

                    # Move to Position -> Positive Error Handling -> Continue Handover
                    if self.robot.move_joint(goal_position): return_flag = True

                    else:

                        # Negative Error Handling -> Stop Handover
                        self.node.get_logger().error('ERROR: An exception occurred during User Moved Error Handling -> Stop Experiment')
                        self.error_stop = True
                        return False

                # Put Object in Place Area -> Place Object -> Stop Handover
                elif self.error_handling_command.fused_command in [PLACE_OBJECT_GIVEN_AREA, PLACE_OBJECT_GIVEN_AREA_POINT_AT]:

                    # Place Object in Area Error Handling -> Stop Handover
                    if not self.placeObject(place_area=self.error_handling_command.area, object_name=object_name):

                        # Negative Error Handling -> Stop Handover
                        self.node.get_logger().error('ERROR: An exception occurred during User Moved Error Handling -> Stop Experiment')
                        self.error_stop = True

                    # Stop Handover
                    return_flag = False

        else:

            # Other Errors -> Stop Handover
            self.node.get_logger().error(f'ERROR: An exception occurred with Untracked Error {handover_error} -> Stop Experiment')
            self.error_stop = True
            return False

        # Clear Error Handling Messages
        self.error_handling_command = None
        self.received_error = None

        # Return Flag -> True: Continue Handover | False: Stop Handover
        return return_flag

    def collaborativeReleaseRoutine(self):

        """ Routine to Help Place the Special Block """

        # Publish Error Message
        error_msg = String()
        error_msg.data = self.HELP_SPECIAL_BLOCK_ERROR_STRING
        self.ttsPub.publish(error_msg)
        self.node.get_logger().warn('ERROR: Help Special Block Error')

        # Wait for Command -> Can Go
        self.wait_for_command = True

        # Wait for Error Handling Command
        while self.error_handling_command is None and not rclpy.ok():
            self.node.get_logger().info_throttle(5, 'Waiting for Error Handling Command')

        # User Have the Special Block -> Open Gripper -> Stop Handover
        if self.error_handling_command.fused_command == CAN_GO:

            self.node.get_logger().info('Object Moved Command -> Retry Place Object')

            # Clear Error Handling Messages
            self.error_handling_command = None
            self.wait_for_command = False

            # Move to Position -> Positive Error Handling -> Continue Handover
            # Release Object | Error -> Return
            self.node.get_logger().info('Open Gripper')
            if not self.robot.move_gripper(GRIPPER_OPEN, self.gripper_enabled): return False
            rclpy.sleep(1)

            # Normal Velocity
            self.setPlanningParameters(velocity_factor=0.2)

            # Move to Home | Error -> Return
            self.node.get_logger().info('Move To Home')
            if not self.robot.move_joint(HOME): return False

            return True

        else:

            # Other Errors -> Stop Handover
            self.node.get_logger().error(f'ERROR: An exception occurred with Incorrect Command {self.error_handling_command.fused_command} -> Stop Experiment')
            self.error_stop = True
            return False

    def enableObstacle(self, obstacle_name, enable=True):

        # Create Bool Message
        msg = Bool()
        msg.data = enable

        if obstacle_name == 'green_block':   self.enableGreenObstaclePub.publish(msg);   self.node.get_logger().warn('Enable Green Obstacle')
        if obstacle_name == 'red_block':     self.enableRedObstaclePub.publish(msg);     self.node.get_logger().warn('Enable Red Obstacle')

        # FIX: Obstacles Disabled
        # if obstacle_name == 'yellow_block':  self.enableYellowObstaclePub.publish(msg);  rospy.logwarn('Enable Yellow Obstacle')
        # if obstacle_name == 'special_block': self.enableSpecialObstaclePub.publish(msg); rospy.logwarn('Enable Special Obstacle')

        # FIX: Disable All Obstacles
        msg.data = False
        if obstacle_name == 'yellow_block': self.enableAllObstaclesPub.publish(msg); self.node.get_logger().warn('Disable All Obstacles')

    def setPlanningParameters(self, velocity_factor, acceleration_factor=0.5):

        self.set_planning_parameters_service.wait_for_service()
        req = PlanningParametersRequest()
        req.velocity_factor = velocity_factor
        req.acceleration_factor = acceleration_factor
        self.set_planning_parameters_service.call(req)

    def run(self):

        """ Run the Experiment """

        # Wait for Experiment Start
        while not self.experiment_started and not rclpy.ok(): self.node.get_logger().info_throttle(5, 'Waiting for Experiment Start')

        # Start Experiment
        self.node.get_logger().info('Start Experiment - Move to Home')
        self.robot.move_joint(HOME)

        # Handover Object
        for object in object_list:

            # Break if Stop Signal Received
            if self.error_stop: break

            if not self.handover(object.name, object.pick_position, object.place_position):

                # Error Handling
                print('ERROR: An exception occurred during Handover')

            # Enable Obstacle
            self.enableObstacle(object.name, True)

        # End of Experiment -> Deactivate All Obstacles
        msg = Bool()
        msg.data = False
        self.node.get_logger().warn('Disable All Obstacles')
        self.enableAllObstaclesPub.publish(msg)

        # Call Set Planning Parameters Service -> Change Planning Velocity
        self.setPlanningParameters(velocity_factor=0.2)

        # Reset Flags
        self.experiment_started = False
        self.error_stop = False

if __name__ == '__main__':

    # Initialize Experiment Manager Node
    exp = ExperimentManager()

    # Run the Experiment
    while not rclpy.ok(): exp.run()
