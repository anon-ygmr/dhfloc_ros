#!/usr/bin/env python
import rospy
import rospkg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from naoqi_bridge_msgs.msg import FloatStamped
import json
import message_filters
from tf.transformations import (
    euler_from_quaternion,
)


class LocalizationResultExporterNode:
    def __init__(self) -> None:
        rospy.init_node("localization_result_exporter_node")
        rospy.loginfo("Localization result exporter node created")

        self.export_file_name = rospy.get_param("~export_file_name")
        self.truth_topic = rospy.get_param("~truth_topic")  # /odometry/filtered
        self.pose_topic = rospy.get_param("~pose_topic", "/robot_pose")  # /robot_pose
        self.comptime_topic = rospy.get_param(
            "~comptime_topic", "/comptime"
        )  # /comptime

        self.sub_truth = message_filters.Subscriber(self.truth_topic, Odometry)
        self.sub_pose = message_filters.Subscriber(self.pose_topic, PoseStamped)
        self.sub_comptime = message_filters.Subscriber(
            self.comptime_topic, FloatStamped
        )

        time_sync_sensors = message_filters.ApproximateTimeSynchronizer(
            [self.sub_truth, self.sub_pose, self.sub_comptime], 100, 0.1
        )
        time_sync_sensors.registerCallback(self.cb_truth_pose)

        self.topicdata = []

    def cb_truth_pose(self, truth_msg, pose_msg, comptime_msg):
        detection_timestamp = truth_msg.header.stamp.to_sec()

        truth = self.extract_odom_msg(truth_msg)
        pose = self.extract_pose_msg(pose_msg)

        comptime = comptime_msg.data

        self.log_data(truth, pose, comptime, detection_timestamp)

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

    def extract_pose_msg(self, pose_msg):
        """Creates a planar pose vector from the pose message.

        Args:
            odom_msg (:obj:`geometry_msgs.msg.PoseStamped`)

        Returns:
            :obj:`list` Containing the x,y position in `m` and the yaw angle in `rad`.
        """

        pose = pose_msg.pose
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

    def log_data(self, truth, pose, comptime, timestamp):
        """Creates one log entry of the given data. Appends it to the log.

        Args:
            truth (:obj:`list`): Ground truth pose data: x,y,heading.
            pose (:obj:`list`): Filtered pose data: x,y,heading.
            comptime (float): Computational time for the given filter step in msecs.
            timestamp (:obj:`genpy.rostime.Time`): Timestamp of the entry in secs.
        """
        self.topicdata.append(
            {"t": timestamp, "truth": truth, "pose": pose, "comptime": comptime}
        )


def save_data(data, filename="topicexport"):
    """Saves data to a file in `json` format.

    Saves the file into dhf_loc/results.

    Args:
        data (:obj:`dict`): Data to be exported.
        filename (:obj:`str`, optional): Name of the file without extension. Defaults to "topicexport".
    """

    # get an instance of RosPack with the default search paths
    rospack = rospkg.RosPack()
    path = rospack.get_path("dhf_loc") + "/assets/results/" + filename + ".json"

    with open(path, "w") as file:
        json.dump({"data": data}, file)


if __name__ == "__main__":
    topicexporter_node = LocalizationResultExporterNode()
    rospy.spin()

    rospy.on_shutdown(
        lambda: save_data(
            topicexporter_node.topicdata, topicexporter_node.export_file_name
        )
    )
