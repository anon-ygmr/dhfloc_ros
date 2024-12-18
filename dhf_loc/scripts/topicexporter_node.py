#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from naoqi_bridge_msgs.msg import FloatStamped
import rospkg
import json
import message_filters
from tf.transformations import (
    euler_from_quaternion,
)


class TopicExporterNode:
    def __init__(self) -> None:
        rospy.init_node("topicexporter_node")
        rospy.loginfo("Topic exporter node created")

        self.pkg_name = rospy.get_param("~pkg_name")
        self.export_file_name = rospy.get_param("~export_file_name")
        self.amcl_topic = rospy.get_param("~amcl_topic", "")
        self.amcl_comptime_topic = rospy.get_param("~amcl_comptime_topic", "")
        self.detection_tolerance = rospy.get_param("~detection_tolerance")

        if self.amcl_topic == "":

            self.scan_topic = rospy.get_param("~scan_topic")
            self.odom_topic = rospy.get_param("~odom_topic")
            self.truth_topic = rospy.get_param("~truth_topic")
            self.sub_scan = message_filters.Subscriber(self.scan_topic, LaserScan)
            self.sub_odom = message_filters.Subscriber(self.odom_topic, Odometry)
            self.sub_truth = message_filters.Subscriber(self.truth_topic, Odometry)
            time_sync_sensors = message_filters.ApproximateTimeSynchronizer(
                [self.sub_scan, self.sub_odom, self.sub_truth], 100, self.detection_tolerance
            )
            time_sync_sensors.registerCallback(self.cb_scan_odom_truth)

            self.prev_odom = None

        else:
            self.sub_amcl = message_filters.Subscriber(
                self.amcl_topic, PoseWithCovarianceStamped
            )
            self.sub_amcl_comptime = message_filters.Subscriber(
                self.amcl_comptime_topic, FloatStamped
            )
            amcl_sync = message_filters.ApproximateTimeSynchronizer(
                [self.sub_amcl, self.sub_amcl_comptime],
                100,
                self.detection_tolerance,
                allow_headerless=True,
            )
            amcl_sync.registerCallback(self.cb_amcl)

        self.topicdata = []

    def cb_scan_odom_truth(self, scan_msg, odom_msg, truth_msg):
        detection_timestamp = scan_msg.header.stamp.to_sec()  # almost the same as odom
        scan = self.extract_scan_msg(scan_msg)
        odom = self.extract_odom_msg(odom_msg)
        truth = self.extract_odom_msg(truth_msg)

        if self.prev_odom is None:
            self.prev_odom = odom
            return

        self.log_data(scan, odom, truth, detection_timestamp)

    def cb_amcl(self, amcl_msg, comptime_msg):
        detection_timestamp = amcl_msg.header.stamp.to_sec()
        comptime = comptime_msg.data
        amcl = self.extract_odom_msg(amcl_msg)

        self.log_data_amcl(amcl, comptime, detection_timestamp)

    def log_data(self, scan, odom, truth, timestamp):
        """Creates one log entry of the given data. Appends it to the log.

        Args:
            scan (:obj:`list`): List of range measurements.
            odom (:obj:`list`): Odometry data: x,y,heading.
            truth (:obj:`list`): Ground truth pose data: x,y,heading.
            timestamp (:obj:`genpy.rostime.Time`): Timestamp of the entry in secs.
        """
        self.topicdata.append(
            {
                "t": timestamp,
                "odom": [
                    round(odom[0], 3),
                    round(odom[1], 3),
                    round(odom[2], 3),
                ],
                "scan": scan,
                "truth": truth
            }
        )

    def log_data_amcl(self, pose, comptime, timestamp):
        self.topicdata.append(
            {
                "t": timestamp,
                "amcl": [
                    round(pose[0], 3),
                    round(pose[1], 3),
                    round(pose[2], 3),
                ],
                "comptime": comptime,
            }
        )

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


def save_data(data, filename="topicexport"):
    """Saves data to a file in `json` format.

    Saves the file into dhf_loc/assets/bag_exports.

    Args:
        data (:obj:`dict`): Data to be exported.
        filename (:obj:`str`, optional): Name of the file without extension. Defaults to "topicexport".
    """

    # get an instance of RosPack with the default search paths
    rospack = rospkg.RosPack()
    path = rospack.get_path("dhf_loc") + "/assets/bag_exports/" + filename + ".json"

    with open(path, "w") as file:
        json.dump({"data": data}, file)


if __name__ == "__main__":
    topicexporter_node = TopicExporterNode()
    rospy.spin()

    rospy.on_shutdown(
        lambda: save_data(
            topicexporter_node.topicdata, topicexporter_node.export_file_name
        )
    )
