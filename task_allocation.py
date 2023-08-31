"""
**************************************************************************
 *   Copyright (C) 2023 Erick Ordaz                                      *
 *   erick.ordazrv@uanl.edu.mx                                           *
 *                                                                       *
 *   Foraging task                                                       *                                                                                
 *   Language: Python                                                    *
 *   Rev: 1.0                                                            *
 *                                                     Fecha: 01/08/23   *
 *************************************************************************
"""

import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
import math
import random
import time


def normalize_angle(angle):
    return angle % (2 * math.pi)


def euclidean_distance(point1, point2):
    distance = 0
    # Itera a través de cada una de las coordenadas de los puntos.
    for i in range(len(point1)):
        # Calcula la suma de las diferencias al cuadrado entre las coordenadas de los dos puntos.
        distance += (point1[i] - point2[i])**2
    # Calcula la raíz cuadrada de la suma de las diferencias al cuadrado para obtener la distancia euclidiana.
    return math.sqrt(distance)


def calculate_distance(ind_i, ind_j, delta, distance_range):
    if delta < distance_range / 2:
        distance = np.hypot(ind_i[0] - ind_j[0], ind_i[1] - ind_j[1])
    else:
        distance = math.inf
    return min(distance, math.inf)


# Function to normalize a value within a range
def normalize(value, min_val, max_val):
    if max_val - min_val == 0:
        return 0
    return (value-min_val) / (max_val-min_val)


# Function to denormalize a value within a range
def denormalize(normalized_value, min_val, max_val):
    return normalized_value * (max_val-min_val) + min_val


def dynamic_model(c, t, u):
    # Define parameters of the robot
    params = {
        'm': 0.38,
        'Im': 0.005,
        'd': 0.02,
        'r': 0.03,
        'R': 0.05
    }
    m, Im, d, r, R = params.values()

    # Define matrices
    M = np.matrix([[m, 0], [0, Im + m * d ** 2]])
    H = np.array([[-m * d * c[5] ** 2], [m * d * c[4] * c[5]]])
    B = np.matrix([[1 / r, 1 / r], [R / r, -R / r]])
    A = np.matrix([[r / 2, r / 2], [r / (2 * R), -r / (2 * R)]])
    Ts = np.matrix([[0.434, 0], [0, 0.434]])
    Ks = np.matrix([[2.745, 0], [0, 2.745]])
    Kl = np.matrix([[1460.2705, 0], [0, 1460.2705]])

    # Calculate velocity
    dxdt = np.concatenate((
        np.asarray(np.matrix([[np.cos(c[3]), -d * np.sin(c[3])], [np.sin(c[3]), d * np.cos(c[3])]]) * np.array(
            [[c[4]], [c[5]]])),
        np.array([[c[4]], [c[5]]]),
        np.linalg.inv(M + B @ np.linalg.inv(Kl) @ Ts @ np.linalg.inv(A)) @ (B @ np.linalg.inv(Kl) @ Ks @ u - (
            H + B @ np.linalg.inv(Kl) @ np.linalg.inv(A) @ np.array([[c[4]], [c[5]]])))
    ), axis=0)

    return np.squeeze(np.asarray(dxdt))


def movement(ci, u):
    # Define initial and final times
    initial_time, final_time = 0, 1
    # Generate a sequence of time samples
    t = np.linspace(initial_time, final_time, 10)
    # Integrate the dynamic model over the given time interval with the given inputs
    c = odeint(dynamic_model, ci, t, args=(u,))

    return c[-1, :]


def foraging(objects, individuals, r_r, o_r, a_r, animation):
    # *[report, ob_ep, ob_ip, individuals_report, optimization_functions] = foraging(objects, individuals, r_r, o_r, a_r, animation)
    cs = np.zeros((individuals, 6))  # Initial individuals states
    c = np.zeros((individuals, 6))  # Individuals states
    report = np.zeros((100000, individuals, 4))  # States report
    # Delivery time, Search time, Collected objects, Distance, Battery
    individuals_report = np.zeros((individuals, 5))
    state_detected = np.zeros((100000, individuals))
    iterations = 0  # Iterations

    # *Visuals
    if animation:
        plt.figure(figsize=(10, 10), dpi=80)
        # ax = plt.gca()  # Nest full (end task)

    # *Objective functions
    optimization_functions = np.zeros((6, 1))
    f1, f2, f3, f4, f5, f6 = 0, 0, 0, 0, 0, 0
    # execution time, energy used, number of members of the swarm, swarm efficiency, task balance, uncollected objects

    # *Parameters enviroment
    white_noise = random.random() * 0.01  # White noise
    area_limits = 10  # Area limit
    nest_radius = 50  # Maximum distance of influence (nest)
    box_radius = 50  # Maximum distance of influence (objects box)

    # *Parameters robots
    desired_voltage = np.zeros((individuals, 2))
    repulsion_voltage, orientation_voltage, attraction_voltage = 2, 2.7, 3.7
    repulsion_radius, orientation_radius, attraction_radius, influence_radius = 0.075 + \
        r_r, 0.075 + o_r, 0.075 + a_r, 3
    out_of_range = np.zeros(individuals)
    explore = np.zeros(individuals)
    explore = [0] * individuals

    # *Walls
    avoid_direction = 0

    # *Battery variables
    battery = [100] * individuals
    battery_discharge = [0.25, 0.5]
    low_battery = 10
    stop = [0] * individuals  # low battery

    # *Parameters robots - objects
    grip_state = np.zeros(individuals)  # Open grip/Close grip
    objects_distances = np.zeros((individuals, objects))
    objects_angles = np.zeros((individuals, objects))

    # *Nest
    nest_arealimits = 0.2
    nest_dot = np.zeros(2) + area_limits * (nest_arealimits / 2)  # Nest dot
    nest_location = [nest_dot[0], nest_dot[1]]  # Nest location (Dotted line)
    # Influence of nest activated by individual
    nest_influence = np.zeros(individuals)

    # *Objects box
    box_center, box_limits = 0.75, 0.2
    objectbox = [box_center * area_limits, box_center * area_limits]

    # *Objects
    objects_location = np.zeros((objects, 2))  # Objects location
    object_in_nest = np.zeros(objects)
    object_in_box = np.ones(objects)
    ob_ip = np.zeros((objects, 2))  # Initial position of objects
    ob_ep = np.zeros((objects, 2))  # Final position of objects

    if objects == 0:
        object_available = np.zeros(1)  # Availability of object to be taken
        grabbed_object = np.zeros(1)  # Objects gripped by individual
    else:
        # Availability of object to be taken
        object_available = np.zeros(objects) + individuals + 1
        # Objects gripped by individual
        grabbed_object = np.zeros(objects) + individuals + 1

    # *Random objects position
    for o in range(objects):
        random_x_position = denormalize(
            random.random(), box_center - (box_limits / 2), box_center + (box_limits / 2))
        random_y_position = denormalize(
            random.random(), box_center - (box_limits / 2), box_center + (box_limits / 2))
        objects_location[o] = [area_limits *
                               random_x_position, area_limits * random_y_position]
        ob_ip[o] = objects_location[o]
        object_available[o] = 1  # Search mode, available object

    # *Initial conditions
    for i in range(individuals):
        if i == 0:
            c[i, :2] = [random.uniform(0, area_limits * 0.25)
                        for _ in range(2)]
        else:
            while True:
                c[i, :2] = [random.uniform(0, area_limits * 0.25)
                            for _ in range(2)]
                if all(math.sqrt((c[i, 0] - c[j, 0]) ** 2 + (c[i, 1] - c[j, 1]) ** 2) > 0.3 for j in range(i)):
                    break
        # Movement, Orientation, Speed, Angular speed
        c[i, 2:] = [0, random.uniform(0, 2 * math.pi), 0, 0]
    dirExp = c[:, 3]

    # *Finish task when nest is full
    while np.mean(object_in_nest) != 1 and iterations < 6000:
        iterations = iterations + 1

        for i in range(individuals):
            desired_voltage[i] = [orientation_voltage + white_noise] * 2

            # *Elements detected
            objects_detected = np.zeros(individuals)
            repulsion_walls = np.zeros(individuals)
            repulsion_detected = np.zeros(individuals)
            orientation_detected = np.zeros(individuals)
            attraction_detected = np.zeros(individuals)
            elements_rx, elements_ry = [], []
            elements_ox, elements_oy = [], []
            elements_ax, elements_ay = [], []

            # *Perception range
            repulsion_range = 3.14159
            orientation_range = 3.14159
            attraction_range = 3.14159
            influence_range = 3.14159
            nest_range = 2*math.pi
            objectbox_range = 2*math.pi

            if np.mean(object_in_box) == 0:
                box_radius = 0

            # Verify each sensor for repulsion of walls
            if out_of_range[i] == 0:
                for w in range(5):
                    avoid_direction = c[i, 3] - \
                        3.83972 if w == 0 else avoid_direction + 1.91986
                    avoid_direction = normalize_angle(avoid_direction)

                    Dir = [math.cos(avoid_direction),
                           math.sin(avoid_direction)]
                    limitX = c[i, 0] + (Dir[0] * repulsion_radius)
                    limitY = c[i, 1] + (Dir[1] * repulsion_radius)

                    # Resulting direction due exploration
                    if limitX > area_limits or limitX < 0 or limitY > area_limits or limitY < 0:
                        dirExp[i] = avoid_direction + \
                            (3 * math.pi / 4) + \
                            (random.uniform(0, 1) * math.pi / 2)
                        repulsion_walls[i] += 1
                        repulsion_walls[i] = normalize_angle(
                            repulsion_walls[i])

            # Angle respect to box
            objectbox_angle = math.atan2(
                objectbox[1] - c[i, 1], objectbox[0] - c[i, 0])
            objectbox_angle = normalize_angle(objectbox_angle)

            # Calculation of influence angles by object box
            ob_Beta = objectbox_angle - c[i, 3]
            ob_Beta = normalize_angle(ob_Beta)

            ob_Gamma = c[i, 3] - objectbox_angle
            ob_Gamma = normalize_angle(ob_Gamma)

            ob_Delta = min(ob_Beta, ob_Gamma)

            # Calculated distance between the robots and the object zone
            objectbox_distance = calculate_distance(
                c[i], objectbox, ob_Delta, objectbox_range)

            # Angle respect to nest
            nest_angle = math.atan2(
                (nest_location[1] - c[i, 1]), (nest_location[0] - c[i, 0]))
            nest_angle = normalize_angle(nest_angle)

            # Calculation of influence angles
            n_Beta = nest_angle - c[i, 3]
            n_Beta = normalize_angle(n_Beta)

            n_Gamma = c[i, 3] - nest_angle
            n_Gamma = normalize_angle(n_Gamma)

            n_Delta = min(n_Beta, n_Gamma)

            # Calculated nest distance
            nest_distance = calculate_distance(
                c[i], nest_location, n_Delta, nest_range)

            for j in range(individuals):
                if i == j:  # It must not be the same
                    continue

                elif i != j:
                    # *Angle of the individual with respect to other members of the swarm
                    neighbors_angle = math.atan2(
                        (c[j, 1] - c[i, 1]), (c[j, 0] - c[i, 0]))
                    neighbors_angle = normalize_angle(neighbors_angle)

                    # *Calculation of angles of repulsion and attraction with respect to other individuals
                    beta = neighbors_angle - c[i, 3]
                    beta = normalize_angle(beta)

                    gamma = c[i, 3] - neighbors_angle
                    gamma = normalize_angle(gamma)

                    delta = min(beta, gamma)

                    # *Calculation of the repulsion distance with respect to other individuals
                    repulsion_distance = calculate_distance(
                        c[i], c[j], delta, repulsion_range)

                    # *Calculation of the attraction distance with respect to other individuals
                    attraction_distance = calculate_distance(
                        c[i], c[j], delta, attraction_range)

                    # *Calculation of the orientation distance with respect to other individuals
                    orientation_distance = calculate_distance(
                        c[i], c[j], delta, orientation_range)

                    # *Count the number of individuals detected in the radius of repulsion, orientation, and attraction
                    if repulsion_distance <= repulsion_radius:
                        elements_rx.append(math.cos(neighbors_angle))
                        elements_ry.append(math.sin(neighbors_angle))
                        repulsion_detected[i] += 1

                    if orientation_radius < attraction_distance <= attraction_radius and repulsion_detected[i] == 0:
                        elements_ax.append(math.cos(neighbors_angle))
                        elements_ay.append(math.sin(neighbors_angle))
                        attraction_detected[i] += 1

                    if repulsion_radius < orientation_distance <= orientation_radius and repulsion_detected[i] == 0:
                        elements_ox.append(math.cos(c[j, 3]))
                        elements_oy.append(math.sin(c[j, 3]))
                        orientation_detected[i] += 1

            for o in range(objects):
                # *Search state, object_available = 1, grip_state = 0
                if object_available[o] == 1 and grip_state[i] == 0:

                    # Angle of objects
                    object_angle = math.atan2(
                        (objects_location[o, 1] - c[i, 1]), (objects_location[o, 0] - c[i, 0]))
                    object_angle = normalize_angle(object_angle)

                    # Calculation of influence angles
                    o_Beta = object_angle - c[i, 3]
                    o_Beta = normalize_angle(o_Beta)

                    o_Gamma = c[i, 3] - object_angle
                    o_Gamma = normalize_angle(o_Gamma)

                    o_Delta = min(o_Beta, o_Gamma)

                    # Calculated influence distance by object
                    object_distance = calculate_distance(
                        c[i], objects_location[o], o_Delta, influence_range)

                    if object_distance <= influence_radius:
                        objects_distances[i, o] = object_distance
                        objects_angles[i, o] = object_angle
                        objects_detected[i] += 1
                    else:
                        objects_distances[i, o] = np.inf
                        objects_angles[i, o] = c[i, 3]

                    object_limit = 0.2  # Distance between objects and robot
                    if battery[i] > low_battery and object_distance <= object_limit:
                        # Influence of nest activated by individual
                        nest_influence[i] = 1
                        grabbed_object[o] = i  # Objects gripped by individual
                        individuals_report[i, 2] += 1  # Collected objects
                        # Object taken by robot, unavailable object
                        object_available[o] = 0
                        grip_state[i] = 1  # Close grip

                # Delivery state, object_available  = 0, grip_state = 1
                elif object_available[o] == 0 and grip_state[i] == 1:

                    objects_distances[i, o] = np.inf
                    objects_angles[i, o] = c[i, 3]

                    # Distance between nest and robot
                    nest_limit = 0.25
                    if nest_distance <= nest_limit:
                        nest_influence[i] = 0
                        objects_location[o] = nest_location
                        nest_dot += np.random.uniform(-0.1, 0.1, size=(2,))
                        nest_location = nest_dot.copy()
                        grip_state[i] = 0  # Open grip
                        grabbed_object[o] = individuals + 1

                    if grabbed_object[o] == individuals + 1:
                        objects_location[o] = nest_location
                    elif grip_state[int(grabbed_object[o])] == 1 and nest_influence[i] == 1 and battery[int(grabbed_object[o])] > 0:
                        objects_location[o] = c[int(grabbed_object[o])][:2]
                        object_available[o] = 0  # Unavailable  object

                if euclidean_distance(nest_location, objects_location[o]) <= 1:
                    object_in_nest[o] = 1
                    object_available[o] = 0
                else:
                    object_in_nest[o] = 0

                if euclidean_distance(objectbox, objects_location[o]) <= 2:
                    object_in_box[o] = 1
                else:
                    object_in_box[o] = 0

                if stop[i] == 1:
                    object_available[o] = 1  # Available object
                    grip_state[i] = 0  # Open grip
                    nest_influence[i] = 0  # Objects gripped by individual

            # Average of detected elements
            if repulsion_detected[i] > 0:
                repulsion_direction = math.atan2(
                    (-np.sum(elements_ry)), (-np.sum(elements_rx)))
                repulsion_direction = normalize_angle(repulsion_direction)

            if orientation_detected[i] > 0:
                orientation_direction = math.atan2(
                    (np.sum(elements_oy)), (np.sum(elements_ox)))
                orientation_direction = normalize_angle(orientation_direction)

            if attraction_detected[i] > 0:
                attraction_direction = math.atan2(
                    (np.sum(elements_ay)), (np.sum(elements_ax)))
                attraction_direction = normalize_angle(attraction_direction)

            if objects_detected[i] > 0:
                objects_distances_index = np.argmin(objects_distances[i, :])
                objects_direction = objects_angles[i, objects_distances_index]
            else:
                objects_direction = c[i, 3]

            # * Behavior Policies
            if repulsion_walls[i] > 0:
                state_detected[iterations, i] = 1

            # Repulsion rules
            if out_of_range[i] == 0 and stop[i] == 0 and repulsion_detected[i] > 0:
                state_detected[iterations, i] = 1
                if nest_influence[i] == 0 and grip_state[i] == 0:
                    if objectbox_distance < box_radius:
                        explore[i] = 1
                        xT = 0.3 * math.cos(c[i, 3]) + 0.4 * math.cos(repulsion_direction) + 0.3 * math.cos(
                            objectbox_angle)
                        yT = 0.3 * math.sin(c[i, 3]) + 0.4 * math.sin(repulsion_direction) + 0.3 * math.sin(
                            objectbox_angle)
                    else:
                        xT = 0.3 * math.cos(c[i, 3]) + 0.4 * math.cos(
                            repulsion_direction) + 0.3 * math.cos(objects_direction)
                        yT = 0.3 * math.sin(c[i, 3]) + 0.4 * math.sin(
                            repulsion_direction) + 0.3 * math.sin(objects_direction)
                        dirExp[i] = repulsion_direction
                elif nest_influence[i] == 1 and grip_state[i] == 1:
                    if nest_distance < nest_radius:
                        xT = 0.3 * \
                            math.cos(
                                c[i, 3]) + 0.4 * math.cos(repulsion_direction) + 0.3 * math.cos(nest_angle)
                        yT = 0.3 * \
                            math.sin(
                                c[i, 3]) + 0.4 * math.sin(repulsion_direction) + 0.3 * math.sin(nest_angle)
                    else:
                        xT = 0.3 * math.cos(c[i, 3]) + 0.4 * math.cos(
                            repulsion_direction) + 0.3 * math.cos(objects_direction)
                        yT = 0.3 * math.sin(c[i, 3]) + 0.4 * math.sin(
                            repulsion_direction) + 0.3 * math.sin(objects_direction)
                        dirExp[i] = repulsion_direction

                desired_voltage[i, :] = [repulsion_voltage + white_noise] * 2
                c[i, 3] = math.atan2(yT, xT)

            if out_of_range[i] == 1:
                if objectbox_distance < nest_distance:
                    xT = math.cos(objectbox_angle)
                    yT = math.sin(objectbox_angle)
                    desired_voltage[i, :] = [
                        orientation_voltage + white_noise] * 2
                    c[i, 3] = math.atan2(yT, xT)
                    if objectbox_distance <= 1:
                        out_of_range[i] = 0
                else:
                    xT = math.cos(nest_angle)
                    yT = math.sin(nest_angle)
                    desired_voltage[i, :] = [
                        orientation_voltage + white_noise] * 2
                    c[i, 3] = math.atan2(yT, xT)
                    if nest_distance <= 1:
                        out_of_range[i] = 0

            elif battery[i] <= low_battery:
                xT = math.cos(nest_angle)
                yT = math.sin(nest_angle)
                desired_voltage[i, :] = [orientation_voltage + white_noise] * 2
                c[i, 3] = math.atan2(yT, xT)

            elif objects_detected[i] > 0:
                state_detected[iterations, i] = 4
                xT = math.cos(objects_direction)
                yT = math.sin(objects_direction)
                desired_voltage[i, :] = [orientation_voltage + white_noise] * 2
                c[i, 3] = math.atan2(yT, xT)

            else:
                # Orientation rules
                if stop[i] == 0 and orientation_detected[i] > 0 and repulsion_detected[i] == 0 and attraction_detected[i] == 0:
                    if nest_influence[i] == 0 and grip_state[i] == 0:
                        if objectbox_distance < box_radius:
                            explore[i] = 1
                            xT = 0.5 * math.cos(c[i, 3]) + \
                                0.5 * math.cos(objectbox_angle)
                            yT = 0.5 * math.sin(c[i, 3]) + \
                                0.5 * math.sin(objectbox_angle)
                        else:
                            state_detected[iterations, i] = 2
                            dirExp[i] = orientation_direction
                            xT = 0.5 * \
                                math.cos(c[i, 3]) + 0.5 * \
                                math.cos(orientation_direction)
                            yT = 0.5 * \
                                math.sin(c[i, 3]) + 0.5 * \
                                math.sin(orientation_direction)
                    elif nest_influence[i] == 1 and grip_state[i] == 1:
                        if nest_distance < nest_radius:
                            state_detected[iterations, i] = 4
                            xT = 0.5 * math.cos(c[i, 3]) + \
                                0.5 * math.cos(nest_angle)
                            yT = 0.5 * math.sin(c[i, 3]) + \
                                0.5 * math.sin(nest_angle)
                        else:
                            state_detected[iterations, i] = 2
                            dirExp[i] = orientation_direction
                            xT = 0.5 * \
                                math.cos(c[i, 3]) + 0.5 * \
                                math.cos(orientation_direction)
                            yT = 0.5 * \
                                math.sin(c[i, 3]) + 0.5 * \
                                math.sin(orientation_direction)
                    desired_voltage[i, :] = [
                        orientation_voltage + white_noise] * 2
                    c[i, 3] = math.atan2(yT, xT)

                # Attraction rules
                if stop[i] == 0 and attraction_detected[i] > 0 and repulsion_detected[i] == 0 and orientation_detected[i] == 0:
                    if nest_influence[i] == 0 and grip_state[i] == 0:
                        if objectbox_distance < box_radius:
                            explore[i] = 1
                            desired_voltage[i, :] = [orientation_voltage] * 2
                            xT = 0.3 * math.cos(c[i, 3]) + 0.4 * math.cos(
                                objectbox_angle) + 0.3 * math.cos(attraction_direction)
                            yT = 0.3 * math.sin(c[i, 3]) + 0.4 * math.sin(
                                objectbox_angle) + 0.3 * math.cos(attraction_direction)
                        else:
                            state_detected[iterations, i] = 3
                            desired_voltage[i, :] = [
                                attraction_voltage + white_noise] * 2
                            dirExp[i] = attraction_direction
                            xT = 0.5 * \
                                math.cos(c[i, 3]) + 0.5 * \
                                math.cos(attraction_direction)
                            yT = 0.5 * \
                                math.sin(c[i, 3]) + 0.5 * \
                                math.sin(attraction_direction)
                    elif nest_influence[i] == 1 and grip_state[i] == 1:
                        if nest_distance < nest_radius:
                            state_detected[iterations, i] = 4
                            desired_voltage[i, :] = [
                                orientation_voltage + white_noise] * 2
                            xT = 0.5 * math.cos(c[i, 3]) + \
                                0.5 * math.cos(nest_angle)
                            yT = 0.5 * math.sin(c[i, 3]) + \
                                0.5 * math.sin(nest_angle)
                        else:
                            state_detected[iterations, i] = 3
                            desired_voltage[i, :] = [
                                attraction_voltage + white_noise] * 2
                            dirExp[i] = attraction_direction
                            xT = 0.5 * \
                                math.cos(c[i, 3]) + 0.5 * \
                                math.cos(attraction_direction)
                            yT = 0.5 * \
                                math.sin(c[i, 3]) + 0.5 * \
                                math.sin(attraction_direction)
                    c[i, 3] = math.atan2(yT, xT)

                # Orientation and Attraction rules
                if stop[i] == 0 and orientation_detected[i] > 0 and attraction_detected[i] > 0 and repulsion_detected[i] == 0:
                    if nest_influence[i] == 0 and grip_state[i] == 0:
                        if objectbox_distance < box_radius:
                            explore[i] = 1
                            xT = 0.5 * math.cos(c[i, 3]) + \
                                0.5 * math.cos(objectbox_angle)
                            yT = 0.5 * math.sin(c[i, 3]) + \
                                0.5 * math.sin(objectbox_angle)
                        else:
                            state_detected[iterations, i] = 3
                            dirExp[i] = math.atan2((math.sin(orientation_direction) + math.sin(attraction_direction)),
                                                   (math.cos(orientation_direction) + math.cos(attraction_direction)))
                            xT = 0.5 * math.cos(c[i, 3]) + 0.25 * math.cos(
                                orientation_direction) + 0.25 * math.cos(attraction_direction)
                            yT = 0.5 * math.sin(c[i, 3]) + 0.25 * math.sin(
                                orientation_direction) + 0.25 * math.sin(attraction_direction)
                    elif nest_influence[i] == 1 and grip_state[i] == 1:
                        if nest_distance < nest_radius:
                            state_detected[iterations, i] = 4
                            xT = 0.5 * math.cos(c[i, 3]) + \
                                0.5 * math.cos(nest_angle)
                            yT = 0.5 * math.sin(c[i, 3]) + \
                                0.5 * math.sin(nest_angle)
                        else:
                            state_detected[iterations, i] = 3
                            dirExp[i] = math.atan2((math.sin(orientation_direction) + math.sin(attraction_direction)),
                                                   (math.cos(orientation_direction) + math.cos(attraction_direction)))
                            xT = 0.5 * math.cos(c[i, 3]) + 0.25 * math.cos(
                                orientation_direction) + 0.25 * math.cos(attraction_direction)
                            yT = 0.5 * math.sin(c[i, 3]) + 0.25 * math.sin(
                                orientation_direction) + 0.25 * math.sin(attraction_direction)
                    desired_voltage[i, :] = [
                        orientation_voltage + white_noise] * 2
                    c[i, 3] = math.atan2(yT, xT)

                # Out of range
                if stop[i] == 0 and attraction_detected[i] == 0 and repulsion_detected[i] == 0 and orientation_detected[i] == 0:
                    if nest_influence[i] == 0 and grip_state[i] == 0:
                        if objectbox_distance < box_radius:
                            explore[i] = 1
                            xT = 0.5 * math.cos(c[i, 3]) + \
                                0.5 * math.cos(objectbox_angle)
                            yT = 0.5 * math.sin(c[i, 3]) + \
                                0.5 * math.sin(objectbox_angle)
                        else:
                            state_detected[iterations, i] = 0
                            xT = 0.5 * math.cos(c[i, 3]) + \
                                0.5 * math.cos(dirExp[i])
                            yT = 0.5 * math.sin(c[i, 3]) + \
                                0.5 * math.sin(dirExp[i])
                    elif nest_influence[i] == 1 and grip_state[i] == 1:
                        if nest_distance < nest_radius:
                            state_detected[iterations, i] = 4
                            xT = 0.5 * math.cos(c[i, 3]) + \
                                0.5 * math.cos(nest_angle)
                            yT = 0.5 * math.sin(c[i, 3]) + \
                                0.5 * math.sin(nest_angle)
                        else:
                            state_detected[iterations, i] = 0
                            xT = 0.5 * math.cos(c[i, 3]) + \
                                0.5 * math.cos(dirExp[i])
                            yT = 0.5 * math.sin(c[i, 3]) + \
                                0.5 * math.sin(dirExp[i])
                    desired_voltage[i, :] = [
                        orientation_voltage + white_noise] * 2
                    c[i, 3] = math.atan2(yT, xT)

            if explore[i] == 1 and objectbox_distance > box_radius and random.random() < 0.1:
                explore[i] = 0
                dirExp[i] = dirExp[i] + \
                    (3 * math.pi / 4) + (random.random() * math.pi / 2)
                dirExp[i] = normalize_angle(dirExp[i])

            report[iterations, i, 0] = c[i, 0]
            report[iterations, i, 1] = c[i, 1]
            report[iterations, i, 2] = c[i, 2]
            report[iterations, i, 3] = c[i, 3]
            individuals_report[i, 3] = c[i, 2]

            c_past = c[i, :]

            if battery[i] > 0:
                battery[i] -= battery_discharge[int(grip_state[i])]
            else:
                stop[i] = 1  # low battery

            if stop[i] == 0:
                cs[i, :] = movement(
                    c[i, :], desired_voltage[i, :].reshape(2, 1))
                c[i, :] = cs[i, :]
            else:
                if battery[i] < 80:
                    # Mientras la batería se está cargando, el robot mantiene su posición
                    c[i, :] = c_past
                    # Solo carga la batería si no ha alcanzado su capacidad máxima
                    if battery[i] <= 100:
                        battery[i] += 1  # La batería se está cargando
                elif battery[i] == 80:
                    stop[i] = 0
                    grip_state[i] = 0

            # Control de límites en los ejes x e y
            if c[i, 0] < 0:
                out_of_range[i] = 1
            elif c[i, 0] > area_limits:
                out_of_range[i] = 1
            elif c[i, 1] < 0:
                out_of_range[i] = 1
            elif c[i, 1] > area_limits:
                out_of_range[i] = 1

            # this avoids an infinite increment of radians
            c[i, 3] = normalize_angle(c[i, 3])

            # Delivery time
            if grip_state[i] == 1:  # Grip close
                individuals_report[i, 0] += 1

        # Simulation
        if animation:

            # plt.figure(figsize=(7, 7), dpi=80)
            ax = plt.gca()
            x = report[iterations, :, 0]
            y = report[iterations, :, 1]
            vx = np.cos(report[iterations, :, 3])
            vy = np.sin(report[iterations, :, 3])

            colors = ['red' if stop[i] == 1 else
                      'dimgray' if state_detected[iterations, i] == 0 else
                      'dimgray' if state_detected[iterations, i] == 1 else
                      'dimgray' if state_detected[iterations, i] == 2 else
                      'dimgray' if state_detected[iterations, i] == 3 else
                      'yellow' if state_detected[iterations, i] == 4 else
                      'dimgray' if state_detected[iterations, i] == 5 else
                      'dimgray' for i in range(individuals)]
            colors_grip = ['green' if grip_state[i] == 0 else
                           'blue' for i in range(individuals)]
            colors_nest = ['green' if nest_influence[i] == 0 else
                           'blue' for i in range(individuals)]
            colors_battrery = ['green' if battery[i] > 60 else
                               'yellow' if 20 < battery[i] <= 60 else
                               'blue' if low_battery < battery[i] <= 20 else
                               'red' for i in range(individuals)]
            colors_limits = ['green' if out_of_range[i] == 0 else
                             'blue' for i in range(individuals)]

            box_rectangle = plt.Rectangle(
                ((box_center - (box_limits / 2)) * area_limits,
                 (box_center - (box_limits / 2)) * area_limits),
                box_limits * area_limits, box_limits * area_limits, color='blue', alpha=0.6, fill=False)

            nest_circle = plt.Circle((nest_location[0], nest_location[1]), nest_radius, color='red', alpha=0.6,
                                     fill=False)
            nest_rectangle = plt.Rectangle((0, 0), nest_arealimits * area_limits, nest_arealimits * area_limits,
                                           color='red',
                                           alpha=0.6, fill=False)

            limit_rectangle = plt.Rectangle(
                (0, 0), area_limits, area_limits, color='blue', alpha=0.6, fill=False)

            plt.cla()
            ax.add_patch(box_rectangle)
            ax.add_patch(nest_circle)
            ax.add_patch(nest_rectangle)
            ax.add_patch(limit_rectangle)

            ax.quiver(x, y, vx, vy, color=colors_battrery)
            for o in range(objects):
                object_color = 'blue' if object_available[o] == 0 else 'green'
                obplt = plt.Circle(
                    (objects_location[o, 0], objects_location[o, 1]), 0.1, color=object_color)
                ax.add_patch(obplt)

            ax.set(xlim=(0 - 1, area_limits + 1), ylim=(
                0 - 1, area_limits + 1), aspect='equal')
            plt.pause(0.000001)

    individuals_report[:, 1] = iterations - individuals_report[:, 0]
    individuals_report[:, 4] = battery[:]
    np.save('report', report)
    np.save('individuals_report', individuals_report)

    f1 = iterations
    f2 = sum(individuals_report[:, 3])
    f3 = individuals
    f4 = sum(individuals_report[:, 1]) / (iterations*individuals)
    f5 = np.std(individuals_report[:, 2])
    f6 = objects - (np.mean(object_in_nest) * objects)
    optimization_functions = np.array([f1, f2, f3, f4, f5, f6]).reshape(6, 1)

    return report, ob_ep, ob_ip, individuals_report, optimization_functions


def main_simulation():
    objects = int(input("Enter the number of objects: "))
    individuals = int(input("Enter the number of individuals: "))
    r_r = float(input("Enter the repulsion radius (m): "))
    o_r = float(input("Enter the orientation radius (m): "))
    a_r = float(input("Enter the attraction radius (m): "))

    animation = input(
        "Do you want to see the animation?? (YES/NO): ").upper() == "YES"

    start_time = time.time()
    [report, ob_ep, ob_ip, individuals_report, optimization_functions] = foraging(
        objects, individuals, r_r, o_r, a_r, animation)
    end_time = time.time()

    print('\nRuntime: ', end_time - start_time)
    np.save('report', report)
    np.save('individuals_report', individuals_report)
    np.save('optimization_functions', optimization_functions)

    return report, ob_ep, ob_ip, individuals_report, optimization_functions


def simulation_mean(replicas):
    #[optimization_functions_report, optimization_functions_mean] = foraging_mean(replicas)
    optimization_functions_report = np.zeros((replicas, 6))
    optimization_functions_mean = np.zeros((replicas, 6))
    animation = False

    objects = int(input("Enter the number of objects: "))
    individuals = int(input("Enter the number of individuals: "))
    r_r = float(input("Enter the repulsion radius (m): "))
    o_r = float(input("Enter the orientation radius (m): "))
    a_r = float(input("Enter the attraction radius (m): "))

    initial_time = time.time()
    percentage = np.linspace(0, 100, replicas + 1)
    print("Progress: ", percentage[0], "%")

    for r in range(replicas):
        [report, ob_ep, ob_ip, individuals_report, optimization_functions] = foraging(
            objects, individuals, r_r, o_r, a_r, animation)
        print("Progress: ", round(percentage[r + 1], 2), "%")
        optimization_functions_report[r, :] = optimization_functions.flatten()

    for r in range(replicas):
        optimization_functions_mean[r, :] = np.mean(
            optimization_functions_report[0:r + 1, :], axis=0)

    final_time = time.time()
    print('\nAverage runtime: ', (final_time - initial_time) / replicas)
    print('Runtime: ', final_time - initial_time)

    return optimization_functions_mean


if __name__ == '__main__':
    menu = """
    Aggregation task in robot swarms \n
    1.- Simple simulation
    2.- Multiple simulations
    3.- Exit
    """

    print(menu)
    while True:
        answer = int(input("Choose an option: "))
        if answer == 1:
            [report, ob_ep, ob_ip, individuals_report,
                optimization_functions] = main_simulation()
            break
        elif answer == 2:
            replicas = int(input("Enter number of replicas: "))
            [optimization_functions_mean] = simulation_mean(replicas)
            break
        elif answer == 3:
            break
        else:
            print("\nError")
            print(menu)
