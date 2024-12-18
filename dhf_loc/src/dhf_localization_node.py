import numpy as np
import json
import time

import rospy
import rospkg
import message_filters
import tf2_ros  # TODO replace with tf_conversions, see tf2 tutorial
from tf.transformations import (
    euler_from_quaternion,
    quaternion_from_euler,
    translation_matrix,
    quaternion_matrix,
    concatenate_matrices,
    translation_from_matrix,
    quaternion_from_matrix,
)

from dhflocalization.gridmap import GridMap
from dhflocalization.filters import EDH, EKF
from dhflocalization.filters.updaters import MEDHUpdater, NAEDHUpdater
from dhflocalization.kinematics import OdometryMotionModel
from dhflocalization.measurement import MeasurementModel, MeasurementProcessor
from dhflocalization.customtypes import StateHypothesis

from nav_msgs.srv import GetMap
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from naoqi_bridge_msgs.msg import FloatStamped


class DhfLocalizationNode:
    """Performs planar localization of a mobile robot on a given map.

    This node assumes a robot with differential drive and a single, horizontally-mounted, fixed 360° LiDAR sensor.
    The localization is performed on an occupancy grid map.
    The algorithm uses the log-homotopy particle flow filter for state estimation (developed by Daum and Huang),
    a chamfer distance based method as measurement model (developed by Dantanarayana and Ranasinghe),
    and the odometry motion model (developed by Thrun et al.).

    Configuration: #TODO add defaults
        ~transform_tolerance (:obj:`float`): Time in seconds to post-date the `~map_frame` -> `odom_frame` transformation.
        ~detection_tolerance (:obj:`float`): Tolerance in seconds between the `LaserScan` and the `Odometry` message
        to be associated together.

        ~medh_particle_number (:obj:`int`): Number of particles to be used in the mean particle flow filter.
        ~naedh_particle_number (:obj:`int`): Number of particles to be used in the n-step analytic particle flow filter.
        ~medh_lambda_number (:obj:`int`): Number of lambda steps to be used in the mean particle flow filter.
        ~naedh_step_number (:obj:`int`): Number of  steps to be used in the n-step analytic particle flow filter.
        ~pseudo_timesteps (:obj:`int`): Number of equal homotopy steps to perform the filter update.

        ~max_ray_number (:obj:`int`): Max number of laser rays to be used. Unused rays are eliminated
        evenly.
        ~laser_range_noise_std (:obj:`float`): Standard deviation of the range readings from the LiDAR.

        ~odometry_alpha_1 (:obj:`float`): Odometry noise to account for error
        in the rotation estimate based on the performed rotation. (deg/deg)
        ~odometry_alpha_2 (:obj:`float`): Odometry noise to account for error
        in the rotation estimate based on the performed translation. (deg/m)
        ~odometry_alpha_3 (:obj:`float`): Odometry noise to account for error
        in the translation estimate based on the performed translation. (m/m)
        ~odometry_alpha_4 (:obj:`float`): Odometry noise to account for error
        in the translation estimate based on the performed rotation. (m/deg)

        ~initial_pose_x (:obj:`float`): Initial robot pose in `x` direction,
        used as the mean in initializing the filter by a Gaussian distribution.
        ~initial_pose_y (:obj:`float`): Initial robot pose in `y` direction,
        used as the mean in initializing the filter by a Gaussian distribution.
        ~initial_pose_heading (:obj:`float`): Initial robot heading,
        used as the mean in initializing the filter by a Gaussian distribution.
        ~initial_cov_x (:obj:`float`): Initial covariance of `x` position (`x*x`),
        used as the covariance in initializing the filter by a Gaussian distribution.
        ~initial_cov_y (:obj:`float`): Initial covariance of `y` position (`y*y`),
        used as the covariance in initializing the filter by a Gaussian distribution.
        ~initial_cov_heading (:obj:`float`): Initial covariance of `heading` (`heading*heading`),
        used as the covariance in initializing the filter by a Gaussian distribution.

        ~edh_type (:obj: `string`): Which variant of the particle flow filter to used.
        Either 'medh', 'naedh' or '', where the latter corresponds to the extended Kalman Filter.

    Subscribers:
        ~scan_topic (:obj:`sensros_msgs.msg.LaserScan`): Range readings from the LiDAR.
        ~odom_topic (:obj:`nav_msgs.msg.Odometry`): The odometry of the differential drive robot.

    Called Services:
        ~static_map_srv (:obj:`nav_msgs.srv.GetMap`): The static map to be localized on.

    Required Transforms:
        ~robot_base_frame -> ~odom_frame.
        ~scan_frame -> ~robot_base_frame.

    Provided Transform:
        ~global_frame -> ~odom_frame

    """

    def __init__(self) -> None:
        rospy.init_node("dhf_localization_node")
        rospy.loginfo("Localization node created")

        # Setting up ROS parameters
        self.scan_topic = rospy.get_param("~scan_topic")
        self.odom_topic = rospy.get_param("~odom_topic")
        self.static_map_srv = rospy.get_param("~static_map_srv")
        self.robot_base_frame = rospy.get_param("~robot_base_frame")
        self.odom_frame = rospy.get_param("~odom_frame")
        self.scan_frame = rospy.get_param("~scan_frame")
        self.global_frame = rospy.get_param("~global_frame")
        self.transform_tolerance = rospy.get_param("~transform_tolerance")
        self.detection_tolerance = rospy.get_param("~detection_tolerance")

        self.medh_particle_number = rospy.get_param("~medh_particle_number")
        self.medh_lambda_number = rospy.get_param("~medh_lambda_number")
        self.naedh_step_number = rospy.get_param("~naedh_step_number")
        self.naedh_particle_number = rospy.get_param("~naedh_particle_number")

        self.max_ray_number = rospy.get_param("~max_ray_number")
        self.laser_range_noise_std = rospy.get_param("~laser_range_noise_std")

        self.odometry_alpha_1 = rospy.get_param("~odometry_alpha_1")
        self.odometry_alpha_2 = rospy.get_param("~odometry_alpha_2")
        self.odometry_alpha_3 = rospy.get_param("~odometry_alpha_3")
        self.odometry_alpha_4 = rospy.get_param("~odometry_alpha_4")

        self.initial_pose_x = rospy.get_param("~initial_pose_x")
        self.initial_pose_y = rospy.get_param("~initial_pose_y")
        self.initial_pose_heading = rospy.get_param("~initial_pose_heading")
        self.initial_cov_x = rospy.get_param("~initial_cov_x")
        self.initial_cov_y = rospy.get_param("~initial_cov_y")
        self.initial_cov_heading = rospy.get_param("~initial_cov_heading")

        self.edh_type = rospy.get_param("~edh_type")

        # Generic attributes
        self.ekf_prior = None
        self.filter_initialized = False
        self.prev_odom = None
        self.gridmap = None
        self.motion_model = None
        self.last_comptime = None

        # Subscribers
        self.sub_scan = message_filters.Subscriber(self.scan_topic, LaserScan)
        self.sub_odom = message_filters.Subscriber(self.odom_topic, Odometry)

        time_sync_sensors = message_filters.ApproximateTimeSynchronizer(
            [self.sub_scan, self.sub_odom],
            100,
            self.detection_tolerance,
        )
        time_sync_sensors.registerCallback(self.cb_scan_odom)

        # Publishers
        self.pub_comptime = rospy.Publisher("/comptime", FloatStamped, queue_size=10)

        # Listeners and broadcasters
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Timer
        self.last_print_time = 0
        self.last_detection_since_epoch = None
        self.print_delay = 5  # seconds
        self.timer = rospy.Timer(rospy.Duration(0.1), self.cb_waiting_timer)

        # Services
        self.srv_get_map = rospy.ServiceProxy(self.static_map_srv, GetMap)
        self.gridmap = self.get_map()  # blocks

    def cb_scan_odom(self, scan_msg, odom_msg):
        """Callback to handle the sensor messages.

        This is only called if the two sensor messages are sufficiently close in time.

        Args:
            scan_msg (:obj:`sensor_msgs.msg.LaserScan`): Laser scan from LiDAR.
            odom_msg (:obj:`nav_msgs.msg.Odometry`): Odom message from the wheel encoders.
        """

        # if self.last_detection_since_epoch is not None:
        #     rospy.loginfo(time.time() - self.last_detection_since_epoch)
        self.last_detection_since_epoch = (
            time.time()
        )  # not simulated time, used for timeout detection

        if self.gridmap is None:
            return

        detection_timestamp = scan_msg.header.stamp.to_sec()  # almost the same as odom
        odom = self.extract_odom_msg(odom_msg)
        scan = self.extract_scan_msg(scan_msg)

        if self.prev_odom is None:
            self.prev_odom = odom
            return

        if not self.filter_initialized:
            self.init_filter()

        control_input = [self.prev_odom, odom]

        comptime_start = time.time()
        measurement = self.process_scan(scan)
        measurement = self.measurement_processer.filter_measurements(measurement)

        # ekf
        ekf_prediction = self.motion_model.propagate(self.ekf_prior, control_input)
        ekf_posterior, _ = self.ekf.update(ekf_prediction, measurement)
        self.ekf_prior = ekf_posterior

        # edh
        if not self.only_ekf:
            prior = self.edh.last_particle_posterior
            prediction = self.motion_model.propagate_particles(prior, control_input)

            prediction_covar = ekf_prediction.covar
            posterior = self.edh.update(
                prediction, prediction_covar, measurement, return_posterior=True
            )
            posterior_mean = posterior.mean()
        else:
            posterior_mean = ekf_posterior.state_vector

        self.broadcast_pose(posterior_mean, detection_timestamp)

        if not self.filter_initialized:
            self.filter_initialized = True

        self.prev_odom = odom

        # calculate and publish comptime
        comptime_end = time.time()
        comptime = (comptime_end - comptime_start) * 1e3  # in ms
        comptime_msg = FloatStamped()
        comptime_msg.header.stamp = rospy.Time.now()
        comptime_msg.data = comptime

        self.last_comptime = comptime
        self.pub_comptime.publish(comptime_msg)

    def broadcast_pose(self, state, timestamp):
        map_to_base_tr = self.transformation_matrix_from_state(state)
        base_to_odom = self.tf_buffer.lookup_transform(
            self.robot_base_frame, self.odom_frame, rospy.Time()
        )
        base_to_odom_tr = self.transformation_matrix_from_msg(base_to_odom)
        map_to_odom_msg = self.msg_from_transformation_matrix(
            map_to_base_tr @ base_to_odom_tr, timestamp
        )

        self.tf_broadcaster.sendTransform(map_to_odom_msg)

    def cb_waiting_timer(self, _):
        """Callback to periodically check for missing messages and services.

        Based on https://github.com/duckietown/dt-core/blob/daffy/packages/deadreckoning/src/deadreckoning_node.py
        """

        need_print = time.time() - self.last_print_time > self.print_delay

        if self.last_detection_since_epoch is not None:
            dt = rospy.get_time() - self.last_detection_since_epoch
            if dt > self.print_delay and need_print:
                rospy.logwarn(
                    "Associated scan and odom message is not received for {} s".format(
                        dt
                    )
                )
        elif self.last_detection_since_epoch is None and need_print:
            rospy.logwarn(
                "No associated scan and odom message is received. Listening on '{}', '{}'. Association tolerance: {} s.".format(
                    self.scan_topic, self.odom_topic, self.transform_tolerance
                )
            )

        if self.gridmap is None and need_print:
            rospy.logwarn(
                "Waiting for map service. Listening on '{}'".format(self.static_map_srv)
            )

        if need_print:
            self.last_print_time = time.time()

    def transformation_matrix_from_state(self, state):
        """Creates a homogeneous transformation matrix.

        Args:
            state (:obj:`(3,1) np.ndarray`): Planar pose of the robot: x,y,heading

        Returns:
            :obj:`(4,4) np.ndarray`: Homogeneous transformation matrix.
        """
        tran = translation_matrix([state[0], state[1], 0])
        rot = quaternion_matrix(quaternion_from_euler(0, 0, state[2]))
        transformation_matrix = concatenate_matrices(tran, rot)
        return transformation_matrix

    def transformation_matrix_from_msg(self, msg):
        """Creates a homogeneous transformation matrix from a message.

        Args:
            msg (:obj:`geometry_msgs.msg.TransformStamped`): Transform message.

        Returns:
            :obj:`(4,4) np.ndarray`: Homogeneous transformation matrix.
        """
        tran = msg.transform.translation
        tran_matrix = translation_matrix([tran.x, tran.y, tran.z])
        rot = msg.transform.rotation
        rot_matrix = quaternion_matrix(
            [
                rot.x,
                rot.y,
                rot.z,
                rot.w,
            ]
        )

        transformation_matrix = concatenate_matrices(tran_matrix, rot_matrix)
        return transformation_matrix

    def msg_from_transformation_matrix(self, tr_matrix, timestamp):
        """Creates a message from a homogeneous transformation matrix.

        The transformation is between the `map` frame and the `odom` frame,
        and used to correct the odometry drift.

        Args:
            tr_matrix (:obj:`(4,4) np.ndarray`): Homogeneous transformation matrix.
            stamp (:obj:`genpy.rostime.Time`): Timestamp of the message in secs.

        Returns:
            :obj:`geometry_msgs.msg.TransformStamped`: Transform message.
        """
        tran = translation_from_matrix(tr_matrix)
        rot = quaternion_from_matrix(tr_matrix)

        msg = TransformStamped()
        msg.header.stamp = rospy.Time(timestamp + self.transform_tolerance)
        msg.header.frame_id = self.global_frame
        msg.child_frame_id = self.odom_frame
        msg.transform.translation.x = tran[0]
        msg.transform.translation.y = tran[1]
        msg.transform.translation.z = tran[2]

        msg.transform.rotation.x = rot[0]
        msg.transform.rotation.y = rot[1]
        msg.transform.rotation.z = rot[2]
        msg.transform.rotation.w = rot[3]

        return msg

    def extract_odom_msg(self, odom_msg):
        """Creates a planar pose vector from the odom message.

        Args:
            odom_msg (:obj:`nav_msgs.msg.Odometry`)

        Returns:
            :obj:`list` Containing the x,y position in `m` and the yaw angle in `rad`.
        """
        pose = odom_msg.pose.pose
        quaternion = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )

        x = round(pose.position.x, 3)
        y = round(pose.position.y, 3)

        euler = euler_from_quaternion(quaternion)
        yaw = round(euler[2], 3)

        return [x, y, yaw]

    def extract_scan_msg(self, scan_msg):
        """Extracts distance readings from the laser scan.

        `inf` readings are substituted by `None`.

        Args:
            scan_msg (:obj:`sensor_msgs.msg.LaserScan`)

        Returns:
            :obj:`list`: List, containing the ranges for each angle.

        """
        ranges = [
            None if elem == float("inf") else round(elem, 3) for elem in scan_msg.ranges
        ]

        return ranges

    def process_scan(self, scan):
        """Appends angles to the range-only scan readings.

        Assumes that every range reading is evenly distributed accross 360 degs.
        Ranges with value `None` or 0.0 are excluded together with their angle.

        Args:
            scan (:obj:`list`): Range readings.

        Returns:
            :obj:`list of (angle,range)`

        """
        angles = np.linspace(0, 2 * np.pi, len(scan))
        angle_range = [
            (angle, range)
            for angle, range in zip(angles, scan)
            if range is not None and range != 0.0
        ]
        return angle_range

    def get_map_from_srv(self):
        """Requests the occupancy grid map.

        Calls the service `static_map` from the `map_server` node which returns a request (:obj:`nav_msgs.srv.GetMap`).

        Returns:
            :obj:`nav_msgs.srv.GetMap`: The message for successful srv. call,
            or `None` if the call has failed.

        """
        rospy.wait_for_service("static_map")
        try:
            map_response = self.srv_get_map()
            rospy.loginfo("Map received")
            return map_response
        except rospy.ServiceException as e:
            rospy.loginfo("Service call failed: %s" % e)
            return None

    def get_map(self):
        """Creates the `GridMap` for the localization.

        Returns:
            :obj:`dhflocalization.gridmap.GridMap`: The internal object for handling the OGM.

        """
        map_response = self.get_map_from_srv()
        map_message = map_response.map
        map_data = np.asarray(map_message.data)

        width = map_message.info.width
        height = map_message.info.height
        resolution = map_message.info.resolution
        map_array = map_data.reshape(height, width)
        map_array = self.convert_occupancy_representation(map_array)

        center_x = -10  # TODO
        center_y = -10  # TODO

        occupancy_grid_map = GridMap(
            np.flip(map_array, 0), resolution, center_x, center_y
        )

        return occupancy_grid_map

    def convert_occupancy_representation(self, map_array):
        """Converts cell occupancy value representation.

        The `map_server` node uses the value `100` to indicate occupied cells,
        `-1` for unknown and `0` for free. This function tranforms these values to
        `0` representing free and unknown cells, and `1` representing the occupied ones.

        Args:
            map_array (:obj:`np.array`): Occ. values of the map cells in a 2D array.

        Returns:
            :obj:`np.array`: Array with the same shape containing the converted values.


        """
        max_val = map_array.max()
        return np.where(map_array < max_val, 0, 1)

    def get_robot_sensor_transform(self):
        """Determines the static transform between the robot and the sensor.

        Only works in 2D, and only handles a statically mounted sensor.

        Returns:
            :obj:`tuple` of :obj:`float`: Translation in x, translation in y, and yaw angle
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                self.robot_base_frame, self.scan_frame, rospy.Time()
            )
            translation = transform.transform.translation
            rotation_quat = transform.transform.rotation
            rotation_euler = euler_from_quaternion(
                (rotation_quat.x, rotation_quat.y, rotation_quat.z, rotation_quat.w)
            )
            return translation.x, translation.y, rotation_euler[2]

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as err:
            rospy.logwarn(err)
            rospy.logwarn(
                "Cannot determine the transform between '{}' and '{}', using default values instead".format(
                    self.robot_base_frame, self.scan_frame
                )
            )
            return 0, 0, 0

    def init_filter(self):
        """Initializes the filters using the ROS parameter server."""
        cfg_random_seed = int(time.time_ns())
        rng = np.random.default_rng(cfg_random_seed)

        self.motion_model = OdometryMotionModel(
            [
                self.odometry_alpha_1,
                self.odometry_alpha_2,
                self.odometry_alpha_3,
                self.odometry_alpha_4,
            ],
            rng=rng,
        )

        (
            robot_sensor_dx,
            _,
            _,
        ) = self.get_robot_sensor_transform()
        measurement_model = MeasurementModel(
            self.gridmap,
            self.laser_range_noise_std,
            robot_sensor_dx,
        )

        cfg_init_gaussian_mean = np.array(
            [self.initial_pose_x, self.initial_pose_y, self.initial_pose_heading]
        )
        cfg_init_gaussian_covar = np.array(
            [
                [self.initial_cov_x, 0, 0],
                [0, self.initial_cov_y, 0],
                [0, 0, self.initial_cov_heading],
            ]
        )

        particle_init_variables = [
            cfg_init_gaussian_mean,
            cfg_init_gaussian_covar,
            rng,
        ]

        self.measurement_processer = MeasurementProcessor(
            max_ray_number=self.max_ray_number
        )
        self.ekf = EKF(measurement_model)
        self.ekf_prior = StateHypothesis(
            state_vector=cfg_init_gaussian_mean, covar=cfg_init_gaussian_covar
        )

        self.only_ekf = False
        if self.edh_type == "medh":
            rospy.loginfo("Using MEDH filter")
            medh_updater = MEDHUpdater(
                measurement_model,
                self.medh_lambda_number,
                "lin",
                self.medh_particle_number,
            )
            self.edh = EDH(medh_updater, *particle_init_variables)
        elif self.edh_type == "naedh":
            rospy.loginfo("Using NAEDH filter")
            naedh_updater = NAEDHUpdater(
                measurement_model, self.naedh_step_number, self.naedh_particle_number
            )
            self.edh = EDH(naedh_updater, *particle_init_variables)
        else:
            rospy.loginfo("Defaulting to EKF filter")
            self.only_ekf = True


if __name__ == "__main__":
    dhf_localization_node = DhfLocalizationNode()
    rospy.spin()
