import cv2
import numpy as np
import matplotlib.pyplot as plt


def parse_points(points_file):
    """
    Parses our hand-annotated 2D/3D points file
    Args:
        points_file: Path to the points file
    Returns:
        A list of 2D points, and a list of 3D points
    """
    points_2d = []
    points_3d = []

    # reading point file
    with open(points_file, "r") as f:
        for line in f:

            # skipping lines that have comments or are empty
            if line.startswith("#") or line.strip() == "":
                continue
            points_all = [int(x) for x in line.strip().split(",")]

            points_3d.append(tuple(points_all[:3]))
            points_2d.append(tuple(points_all[3:]))

    return points_2d, points_3d


def plot_2d_points(points_2d, image, plot_labels=False):
    """
    Plots 2D points on an image with labels
    Args:
        points_2d: List of 2D points
        image: Path to the image
        plot_labels optional
    Returns:
        Plot of image with point set
    """
    NEON_GREEN = (171, 255, 0)
    BRIGHT_ORANGE = (49, 110, 255)

    # reading the image
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    img = np.stack((img,) * 3, axis=-1)

    # plotting points for debugging and vizualisation
    for i, point in enumerate(points_2d):
        cv2.circle(img, (int(point[0]), int(point[1])), 20, NEON_GREEN, -1)

        if plot_labels:
            cv2.putText(
                img,
                str(i),
                (int(point[0]), int(point[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,
                BRIGHT_ORANGE,
                20,
            )

    return img


def plot_compare_2d(points_2d_1, points_2d_2, image):
    """
    Compares 2D points and reprojected values on an image with labels
    Args:
        points_2d_1: List of 2D points
        points_2d_2: Another List of 2D points
        image: Path to the image
    Returns:
        Plot of image with both point sets
    """
    TOMATO_RED = (132, 132, 244)

    temp = plot_2d_points(points_2d_1, image, plot_labels=False)

    # plotting points for debugging and vizualisation
    for i, point in enumerate(points_2d_2):
        cv2.drawMarker(
            temp,
            (int(point[0]), int(point[1])),
            TOMATO_RED,
            markerType=cv2.MARKER_CROSS,
            markerSize=50,
            thickness=20,
        )

    return temp
