"""Shared main() body for the package's nodes: spin until shut down, and treat
every way launch can stop us (Ctrl-C, SIGINT handled by rclpy, or the context
going invalid mid-wait - an rclpy Humble race) as a clean exit."""
import rclpy
from rclpy.executors import ExternalShutdownException


def spin_node(node_cls):
    rclpy.init()
    try:
        rclpy.spin(node_cls())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:  # RCLError "context is not valid" when shutdown races the wait set
        if rclpy.ok():
            raise
    finally:
        rclpy.try_shutdown()
