# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Import needed variables                                                                                                                                                     #
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

import random
import math
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

import threading
import time

import keras
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import pandas as pd

"""# **Physics Simulation Environment**"""

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#  Abbreviate math.cos, math.sin etc. to just cos and sin for clarity:                                                                                                       #
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

def cos(x):
  return math.cos(x)

def sin(x):
  return math.sin(x)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#  Construct the Rocket State Vector and Desired Rocket State Vector:                                                                                                        #
#                                                                                                                                                                            #
#  The state vector describes the orientation, position etc. of the rocket in the physical space which it is in.                                                             #
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

x_g = 0                                 # The x coordinate of the center of gravity (meters)
y_g = 0                                 # The y coordinate of the center of gravity (meters)
z_g = 5000                              # The z coordinate of the center of gravity (meters)
V_x = 0                                 # The x axis velocity of the rocket (meters/second)
V_y = 0                                 # The y axis velocity of the rocket (meters/second)
V_z = -270                                # The z axis velocity of the rocket (meters/second)
V_theta_S = math.pi/2             # The angle positioning of the rocket on the theta s plane (radians)
V_theta_D = 0             # The angle positioning of the rocket on the theta d plane (radians)
R_theta_S = 0.00                           # The angular velocity of the rocket on the theta s plane (radians/second)
R_theta_D = 0                           # The angular velocity of the rocket on the theta d plane (radians/second)
m_d = 235000                             # The dry mass of the rocket weighing structural mass, no fuel (kilograms)
m_w = 120000                           # The wet mass of the rocket weighing fuel mass (kilograms)

def construct_state_vector(x_g, y_g, z_g, V_x, V_y, V_z, V_theta_S, V_theta_D, R_theta_S, R_theta_D, m_d, m_w):
  rocket_state_vector = [
      x_g,
      y_g,
      z_g,
      V_x,
      V_y,
      V_z,
      V_theta_S,
      V_theta_D,
      R_theta_S,
      R_theta_D,
      m_d,
      m_w
  ]

  return rocket_state_vector

rocket_state_vector = construct_state_vector(
    x_g,
    y_g,
    z_g,
    V_x,
    V_y,
    V_z,
    V_theta_S,
    V_theta_D,
    R_theta_S,
    R_theta_D,
    m_d,
    m_w
)

desired_state_vector = construct_state_vector(
    -50,
    -50,
    0,
    0,
    0,
    0,
    2 * (math.pi)/4,
    0,
    0,
    0,
    m_d,
    m_w
)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#  Construct the Rocket Attribute Vector:                                                                                                                                    #
#                                                                                                                                                                            #
#  This is a vector representing the physical and structural characteristrics of the rocket. This could be the rocket's engine thrust, height, etc.                          #
#  This vector will be useful when calculating the implications of certian throttle values on the physical state of the rocket, as throttle is dependent on thrust capacity. #
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

T_m = 6900000                           # Maximum thrust of the rocket engine (kilogram * meters/second^2)
G_m = 71000                            # Maximum thrust of the cold gas thrusters (kilogram * meters/second^2)
L_1 = 5                                # The length along the body of the rocket between the rocket's engine and the rocket's center of gravity (meters)
L_2 = 45                                # The length along the body of the rocket between the rocket's center of gravity and the rocket's cold gas thrusters (meters)
FB = 1950                               # The rate at which the engine burns fuel at full throttle (kilograms/second)

def construct_attribute_vector(T_m, G_m, L_1, L_2, FB):
  rocket_attribute_vector = [
      T_m,
      G_m,
      L_1,
      L_2,
      FB
  ]

rocket_attribute_vector = [
    T_m,
    G_m,
    L_1,
    L_2,
    FB
]

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#  Construct the Rocket Input Vector:                                                                                                                                        #
#                                                                                                                                                                            #
#  This is a vector representing the controls of the rocket such as the engine or the cold gas thrusters that the control algorithm directly operates.                       #
#  This is the only vector which the algorithm can edit. The idea is tha the control algorithm edits this array which translates into consequences in the real world.        #
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

T_h = 0.1                               # Throttle of the rocket engine (proportion)
E_theta_S = 1 * (math.pi)/2             # Gimbal of the rocket engine on the theta S plane (radians)
E_theta_D = 0                           # Gimbal of the rocket engine of the theta D plane (radians)
Th_GCW_theta_S = 0.000                  # Throttle of the theta S plane clockwise cold gas thruster (proportion)
Th_GCC_theta_S = 0.000                  # Throttle of the theta S plane counter clockwise cold gas thruster (proportion)
Th_GCW_theta_D = 0.000                  # Throttle of the theta D plane clockwise cold gas thruster (proportion)
Th_GCC_theta_D = 0.000                  # Throttle of the theta D plane counter clockwise cold gas thruster (proportion)

def construct_input_vector(T_h, E_theta_S, E_theta_D, Th_GCW_theta_S, Th_GCC_theta_S, Th_GCW_theta_D, Th_GCC_theta_D):
  rocket_input_vector = [
      T_h,
      E_theta_S,
      E_theta_D,
      Th_GCW_theta_S,
      Th_GCC_theta_S,
      Th_GCW_theta_D,
      Th_GCC_theta_D
  ]

  return rocket_input_vector

rocket_input_vector = construct_input_vector(
    T_h,
    E_theta_S,
    E_theta_D,
    Th_GCW_theta_S,
    Th_GCC_theta_S,
    Th_GCW_theta_D,
    Th_GCC_theta_D
)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#  Construct the Physics Simulator:                                                                                                                                          #
#                                                                                                                                                                            #
#  This will be a physics simulation function that updates all of the arrays (such as the rocket state vector) every defined increment of time.                              #
#  It baiscally simulates physics and updates all of the variables so the simulation is realistic.                                                                           #
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

d_t = 0.1                              # increments by which simulation happens, delta T
g = -3.7                               # gravitational costant

def simulate_physical_system(rocket_state_vector, rocket_attribute_vector, rocket_input_vector, g, d_t):

  # import control prints

  global R_theta_D_PRIME_RECORD
  global V_x_PRIME_RECORD
  global V_y_PRIME_RECORD

  # Name the variables from the vectors inside of the function for clarity

  x_g = rocket_state_vector[0]
  y_g = rocket_state_vector[1]
  z_g = rocket_state_vector[2]
  V_x = rocket_state_vector[3]
  V_y = rocket_state_vector[4]
  V_z = rocket_state_vector[5]
  V_theta_S = rocket_state_vector[6]
  V_theta_D = rocket_state_vector[7]
  R_theta_S = rocket_state_vector[8]
  R_theta_D = rocket_state_vector[9]
  m_d = rocket_state_vector[10]
  m_w = rocket_state_vector[11]

  T_m = rocket_attribute_vector[0]
  G_m = rocket_attribute_vector[1]
  L_1 = rocket_attribute_vector[2]
  L_2 = rocket_attribute_vector[3]
  FB = rocket_attribute_vector[4]

  T_h = rocket_input_vector[0]
  E_theta_S = rocket_input_vector[1]
  E_theta_D = rocket_input_vector[2]
  Th_GCW_theta_S = rocket_input_vector[3]
  Th_GCC_theta_S = rocket_input_vector[4]
  Th_GCW_theta_D = rocket_input_vector[5]
  Th_GCC_theta_D = rocket_input_vector[6]

  # If the rocket has no fuel left, the engine will not turn on.

  if(m_w < 0):
    T_h = 0

  # Implement Linear Acceleration Equations

  V_x_PRIME = ( cos(V_theta_D) * cos(V_theta_S) * sin(E_theta_S) * T_m * T_h )/(m_d + m_w)
  V_y_PRIME = ( sin(V_theta_D) * cos(V_theta_S) * sin(E_theta_S) * T_m * T_h )/(m_d + m_w)
  V_z_PRIME = ( sin(V_theta_S) * sin(E_theta_S) * T_m * T_h )/(m_d + m_w)

  # Implment Angular Acceleration Equations

  R_theta_S_PRIME =  ( 2 * L_2 * G_m * ( Th_GCC_theta_S - Th_GCW_theta_S ) + 2 * L_1 * T_m * ( cos(E_theta_S) * sin (E_theta_D) * T_h) )/( ( m_d + m_w ) * ( L_1 * L_1 + L_2 * L_2 ) )
  R_theta_D_PRIME =  ( 2 * L_2 * G_m * ( Th_GCC_theta_D - Th_GCW_theta_D ) + 2 * L_1 * T_m * ( cos(E_theta_S) * cos (E_theta_D) * T_h) )/( ( m_d + m_w ) * ( L_1 * L_1 + L_2 * L_2 ) )

  # Translate Velocities into Positions (1st Degree Derivative => 0th Degree Derivative)

  x_g += d_t * V_x
  y_g += d_t * V_y
  z_g += d_t * V_z
  V_theta_S += d_t * R_theta_S
  V_theta_D += d_t * R_theta_D

  # Translate Accelerations into Velocities (2nd Degree Derivative => 1st Degree Derivative)

  V_x += d_t * V_x_PRIME
  V_y += d_t * V_y_PRIME
  V_z += d_t * V_z_PRIME + d_t * (g)
  R_theta_S += d_t * R_theta_S_PRIME
  R_theta_D += d_t * R_theta_D_PRIME

  # Deduct from wet mass

  m_w -= d_t * FB * T_h

  # Update Arrays with inputted variables

  rocket_state_vector[0] = round(x_g, 5)
  rocket_state_vector[1] = round(y_g, 5)
  rocket_state_vector[2] = round(z_g, 5)
  rocket_state_vector[3] = round(V_x, 5)
  rocket_state_vector[4] = round(V_y, 5)
  rocket_state_vector[5] = round(V_z, 5)
  rocket_state_vector[6] = round(V_theta_S, 5)
  rocket_state_vector[7] = round(V_theta_D, 5)
  rocket_state_vector[8] = round(R_theta_S, 5)
  rocket_state_vector[9] = round(R_theta_D, 5)
  rocket_state_vector[10] = round(m_d, 5)
  rocket_state_vector[11] = round(m_w, 5)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Implementation of Search Space Creation Algorithm.                                                                                                                                                                                                      #
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Create a search space given the current situation of the rocket to create the scenario tree of the model predictive controller

def search_space(rocket_input_vector, time_future):

  T_h = rocket_input_vector[0]
  E_theta_S = rocket_input_vector[1]
  E_theta_D = rocket_input_vector[2]
  Th_GCW_theta_S = rocket_input_vector[3]
  Th_GCC_theta_S = rocket_input_vector[4]
  Th_GCW_theta_D = rocket_input_vector[5]
  Th_GCC_theta_D = rocket_input_vector[6]

  scenarios = []

  if(time_future%6 == 1 or time_future%6 == 5):
    scenarios.append([T_h, E_theta_S, E_theta_D, 1, 0, 0, 0])
    scenarios.append([T_h, E_theta_S, E_theta_D, 0, 1, 0, 0])
    scenarios.append([T_h, E_theta_S, E_theta_D, 0, 0, 1, 0])
    scenarios.append([T_h, E_theta_S, E_theta_D, 0, 0, 0, 1])

  else:
    # Calculate different values for T_h_Switch
    T_h_Switch = []
    for i in range(-1, 2, 1):
      T_h_NEXT = round(T_h + 0.1 * i, 5)
      if(i != 0 and T_h_NEXT >= 0.0 and T_h_NEXT <= 1.0):
        T_h_Switch.append(T_h_NEXT)

    # Calculate different values for E_theta_S_Switch
    E_theta_S_Switch = []
    for j in range(-1, 2, 1):
      E_theta_S_NEXT = round(E_theta_S + math.pi/20 * j, 5)
      if(j != 0 and E_theta_S_NEXT >= (math.pi/2 -0.27) and E_theta_S_NEXT <= (math.pi/2 + 0.27)):
        E_theta_S_Switch.append(E_theta_S_NEXT)

    # Calculate different values for E_theta_D_Switch
    E_theta_D_Switch = []
    for k in range(-1, 2, 1):
      E_theta_D_NEXT = round((E_theta_D + math.pi/5 * k)%(2 * math.pi), 5)
      if(k != 0):
        E_theta_D_Switch.append(E_theta_D_NEXT)

    for a in T_h_Switch:
      for b in E_theta_S_Switch:
        for c in E_theta_D_Switch:
          scenarios.append([a, b, c, Th_GCW_theta_S, Th_GCC_theta_S, Th_GCW_theta_D, Th_GCC_theta_D])

  return scenarios

"""# **Convolutional Neural Network**"""

# Load In Pretrained Model

model = keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/ISEF/Safe Landing Spot AI/my_model_2')

# Define Image Functions

def box(x, y, w, h, img):
  image = np.asarray(img).copy()
  x = int(x)
  y = int(y)
  w = int(w)
  h = int(h)
  for i in range(x, x+w, 1):
    for j in range(0, 5, 1):
      image[y+j-1][i] = (255, 0, 0)
  for i in range(x, x+w, 1):
    for j in range(0, 5, 1):
      image[y+h-j-1][i] = (255, 0, 0)
  for i in range(y, y+h, 1):
    for j in range(0, 5, 1):
      image[i][x+j-1] = (255, 0, 0)
  for i in range(y, y+h, 1):
    for j in range(0, 5, 1):
      image[i][x+w-j-1] = (255, 0, 0)
  return image

def boxGreen(x, y, w, h, img):
  image = np.asarray(img).copy()
  x = int(x)
  y = int(y)
  w = int(w)
  h = int(h)
  for i in range(x, x+w, 1):
    for j in range(0, 2, 1):
      image[y+j-1][i] = (0, 255, 0)
  for i in range(x, x+w, 1):
    for j in range(0, 2, 1):
      image[y+h-j-1][i] = (0, 255, 0)
  for i in range(y, y+h, 1):
    for j in range(0, 2, 1):
      image[i][x+j-1] = (0, 255, 0)
  for i in range(y, y+h, 1):
    for j in range(0, 2, 1):
      image[i][x+w-j-1] = (0, 255, 0)
  return image

def boxYellow(x, y, w, h, img):
  image = np.asarray(img).copy()
  x = int(x)
  y = int(y)
  w = int(w)
  h = int(h)
  for i in range(x, x+w, 1):
    for j in range(0, 5, 1):
      image[y+j-1][i] = (255, 255, 0)
  for i in range(x, x+w, 1):
    for j in range(0, 5, 1):
      image[y+h-j-1][i] = (255, 255, 0)
  for i in range(y, y+h, 1):
    for j in range(0, 5, 1):
      image[i][x+j-1] = (255, 255, 0)
  for i in range(y, y+h, 1):
    for j in range(0, 5, 1):
      image[i][x+w-j-1] = (255, 255, 0)
  return image

def crossHeir(x, y, w, h, img):
  image = np.asarray(img).copy()
  x = int(x)
  y = int(y)
  w = int(w)
  h = int(h)
  for i in range(y-int(h/2), y+int(h/2), 1):
    for j in range(x-2, x+2, 1):
      image[i][j] = (255, 255, 0)
  for i in range(x-int(w/2), x+int(w/2), 1):
    for j in range(y-2, y+2, 1):
      image[j][i] = (255, 255, 0)
  return image

def snip(x, y, w, h, image):
  x = int(x)
  y = int(y)
  w = int(w)
  h = int(h)
  snippet = []
  for i in range(y, y+h-1, 1):
    snippet.append(image[i][x:x+w])
  return np.asarray(snippet)

def pathImage(num):
  path = "/content/drive/MyDrive/Artificial Intelligence/Colab Files/ET Landing/render/render"
  numLen = len(str(num));
  zeroNum = 4-numLen;
  zeroPrefix = ""
  for j in range(0, zeroNum, 1):
    zeroPrefix += "0"
  path = path + zeroPrefix + str(num) + ".png"
  return path

# Create a Function that Takes in an Image, returns optimal Landing Coordinates using CNN

def landing_coords(currentX, currentY, imagePath):

  image = np.asarray(Image.open(imagePath)).copy()

  boxArray = []

  def filterLarge():

    for y in range(0, image.shape[0]-100, 20):
      for x in range(0, image.shape[1]-100, 20):
        snippet = []
        snippet.append(np.asarray(Image.fromarray(snip(x, y, 100, 100, image)).resize((55, 43))))
        snippet = np.asarray(snippet)
        curConf = model.predict(snippet)[0][0]
        if(curConf >= 0.90):
          boxArray.append([x, y, 5, 5])

  def filterHorizontal():

    for y in range(0, image.shape[0]-50, 30):
      for x in range(0, image.shape[1]-100, 30):
        snippet = []
        snippet.append(np.asarray(Image.fromarray(snip(x, y, 100, 50, image)).resize((55, 43))))
        snippet = np.asarray(snippet)
        curConf = model.predict(snippet)[0][0]
        if(curConf >= 0.90):
          boxArray.append([x+50, y+25, 5, 5])

  def filterVertical():

    for y in range(0, image.shape[0]-100, 30):
      for x in range(0, image.shape[1]-50, 30):
        snippet = []
        snippet.append(np.asarray(Image.fromarray(snip(x, y, 50, 100, image)).resize((55, 43))))
        snippet = np.asarray(snippet)
        curConf = model.predict(snippet)[0][0]
        if(curConf >= 0.90):
          boxArray.append([x+25, y+50, 5, 5])

  def filterSmall():

    for y in range(0, image.shape[0]-50, 30):
      for x in range(0, image.shape[1]-50, 30):
        snippet = []
        snippet.append(np.asarray(Image.fromarray(snip(x, y, 50, 50, image)).resize((55, 43))))
        snippet = np.asarray(snippet)
        curConf = model.predict(snippet)[0][0]
        if(curConf >= 0.90):
          boxArray.append([x+25, y+25, 5, 5])

  largeThread = threading.Thread(target=filterLarge, args=());
  horizontalThread = threading.Thread(target=filterHorizontal, args=());
  verticalThread = threading.Thread(target=filterVertical, args=());
  smallThread = threading.Thread(target=filterSmall, args=());

  largeThread.start()
  horizontalThread.start()
  verticalThread.start()
  smallThread.start()

  largeThread.join()
  horizontalThread.join()
  verticalThread.join()
  smallThread.join()

  rocketRadius = 120
  selectDistance = 1000
  selectX = 0
  selectY = 0
  for y in range(0, len(image), 20):
    for x in range(0, len(image[0]), 20):
      minDistance = 1000
      for i in boxArray:
        y1 = i[1]
        x1 = i[0]
        distance = math.sqrt((y-y1) * (y-y1) + (x-x1) * (x-x1))
        if(distance < minDistance):
          minDistance = distance
      if(minDistance+20 > rocketRadius):
        image = boxGreen(x, y, 20, 20, image).copy()
        midDistance = math.sqrt((len(image)/2-y) * (len(image)/2-y) + (len(image[0])/2-x) * (len(image[0])/2-x))
        if(midDistance < selectDistance):
          selectDistance = midDistance
          selectX = x
          selectY = y

  image = crossHeir(len(image[0])/2, len(image)/2, 50, 50, image).copy()

  minDistance = 1000
  minInd = 0
  counter = 0
  for i in boxArray:
    y1 = i[1]
    x1 = i[0]
    distance = math.sqrt((len(image[1])/2-y1) * (len(image[1])/2-y1) + (len(image[0])/2-x1) * (len(image[0])/2-x1))
    if(distance < minDistance):
      minDistance = distance
      minInd = counter
    counter += 1

  image = boxYellow(selectX, selectY, 20, 20, image).copy()
  plt.imshow(image)

  return [(selectX - len(image[0])/2 + currentX) , (selectY - len(image)/2 + currentY)]

landing_coords(0, 0, "/content/drive/MyDrive/Artificial Intelligence/Colab Files/ET Landing/render/render0025.png")

"""# **Model Predictive Controller**"""

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Implementation of a Cost Function to Evaluate Rocket Situation                                                                                                                                                                                          #
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Cost function is a squared error function that evaluates each of the parameters by taking the difference between current state and desired state, squaring it and multiplying by weight.
def cost_function(rocket_state_vector, desired_state_vector, do_print):

  totalCost = 0
  weights = [14, 14, 1, 190, 190, 450, 40000, 0, 400000, 13, 0, 0]

  if(rocket_state_vector[2] < 1000):
    weights = [8, 8, 0.1, 80, 80, 20, 800, 0, 3000, 1.3, 0, 0]

  if(rocket_state_vector[2] < 1.5):
    weights = [0.2, 0.2, 0.001, 0.5, 0.5, 2, 8, 0, 30, 0.013, 0, 0]

  for i in range(0, len(rocket_state_vector), 1):

    x = rocket_state_vector[i] - desired_state_vector[i]

    if(i == 0 or i == 1):
      totalCost += weights[i] * (max(20 * math.sqrt(x * x), x*x) + x*x)/2
    else:
      totalCost += weights[i] * (x) * (x)

  if(rocket_state_vector[5] >= 0):
    totalCost += 1000

  if(do_print == True):
    print("Total Cost = " + str(totalCost))
    print("Cost from x_g = " + str(weights[0] * (rocket_state_vector[0] - desired_state_vector[0]) * (rocket_state_vector[0] - desired_state_vector[0])))
    print("Cost from y_g = " + str(weights[1] * (rocket_state_vector[1] - desired_state_vector[1]) * (rocket_state_vector[1] - desired_state_vector[1])))
    print("Cost from z_g = " + str(weights[2] * (rocket_state_vector[2] - desired_state_vector[2]) * (rocket_state_vector[2] - desired_state_vector[2])))
    print("Cost from V_x = " + str(weights[3] * (rocket_state_vector[3] - desired_state_vector[3]) * (rocket_state_vector[3] - desired_state_vector[3])))
    print("Cost from V_y = " + str(weights[4] * (rocket_state_vector[4] - desired_state_vector[4]) * (rocket_state_vector[4] - desired_state_vector[4])))
    print("Cost from V_z = " + str(weights[5] * (rocket_state_vector[5] - desired_state_vector[5]) * (rocket_state_vector[5] - desired_state_vector[5])))
    print("Cost from V_theta_S = " + str(weights[6] * (rocket_state_vector[6] - desired_state_vector[6]) * (rocket_state_vector[6] - desired_state_vector[6])))
    print("Cost from V_theta_D = " + str(weights[7] * (rocket_state_vector[7] - desired_state_vector[7]) * (rocket_state_vector[7] - desired_state_vector[7])))
    print("Cost from R_theta_S = " + str(weights[8] * (rocket_state_vector[8] - desired_state_vector[8]) * (rocket_state_vector[8] - desired_state_vector[8])))
    print("Cost from R_theta_D = " + str(weights[9] * (rocket_state_vector[9] - desired_state_vector[9]) * (rocket_state_vector[9] - desired_state_vector[9])))
    print("Cost from m_d = " + str(weights[10] * (rocket_state_vector[10] - desired_state_vector[10]) * (rocket_state_vector[10] - desired_state_vector[10])))
    print("Cost from m_W = " + str(weights[11] * (rocket_state_vector[11] - desired_state_vector[11]) * (rocket_state_vector[11] - desired_state_vector[11])))

  return totalCost

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Implementation of Reference Point Placing Algorithm                                                                                                                                                                                          #
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# The reference point placing algorithm gives a reference point for the model predictive control algorithm to approach as a desired value for its angular orientation
def generate_reference(rocket_state_vector, desired_state_vector):

  reference = desired_state_vector.copy()

  reference[6] = (desired_state_vector[6] + rocket_state_vector[6])/2
  reference[7] = (desired_state_vector[7] + rocket_state_vector[7])/2
  reference[8] = (desired_state_vector[8] + rocket_state_vector[8])/2
  reference[9] = (desired_state_vector[9] + rocket_state_vector[9])/2

  print("REFERENCE = " + str(reference))

  return reference

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Implementation of Main Model Predictive Control Algorithm                                                                                                                                                                                               #
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

def model_predictive_control(rocket_state_vector, rocket_input_vector, desired_state_vector, g, d_t, sim_time):

  initiation_time = time.time()
  time_horizon = sim_time + 6
  path = [[rocket_state_vector, rocket_input_vector]]
  optimal_path = [[rocket_state_vector, rocket_input_vector]]
  optimal_path_cost = -1;

  leaves = []

  def scenario_tree(current_path, rocket_input_vector, future_time, g, d_t):

    nonlocal time_horizon
    nonlocal optimal_path
    nonlocal optimal_path_cost

    if(future_time == time_horizon):
      end_cost = cost_function(current_path[-1][0], desired_state_vector, False)
      if(end_cost < optimal_path_cost or optimal_path_cost == -1):
        optimal_path = current_path
        optimal_path_cost = end_cost

      return

    options = search_space(rocket_input_vector, future_time)

    if(future_time%6 == 1 or future_time%6 == 2 or future_time%6 == 3):
      threads = []
      for option in options:
        next_path = current_path.copy()
        next_state = current_path[-1][0].copy()
        simulate_physical_system(next_state, rocket_attribute_vector, option, g, d_t * 5)
        next_path.append([next_state, option.copy()])
        threads.append(threading.Thread(target=scenario_tree, args=(next_path, option.copy(), future_time+1, g, d_t)))
        threads[-1].start()

      for thread in threads:
        thread.join()

      return

    else:
      for option in options:
        next_path = current_path.copy()
        next_state = current_path[-1][0].copy()
        simulate_physical_system(next_state, rocket_attribute_vector, option.copy(), g, d_t * 5)
        next_path.append([next_state, option])
        scenario_tree(next_path, option, future_time+1, g, d_t)

      return

  scenario_tree(path, rocket_input_vector, sim_time, g, d_t)

  rocket_input_vector = optimal_path[1][1]
  completion_time = time.time()

  print("")
  print("The Best Path Is: ")
  for i in optimal_path:
    print(str(i))
  print("Cost of Path: " + str(cost_function(optimal_path[-1][0], desired_state_vector, False)))
  print("Action that will be taken = " + str(rocket_input_vector))
  print("Time Taken to Run = " + str(completion_time - initiation_time))
  if(sim_time%6 == 1 or sim_time%6 == 5):
    print("COLD GAS THRUSTER ACTION")
  else:
    print("ENGINE ACTION")

  return rocket_input_vector

"""# **Run Landing Simultaion**"""

# Designate Arrays to keep track of the variables over time to observe rocket actions during flight
rocket_state_vector_RECORD = []
desired_state_vector_RECORD = []
rocket_attribute_vector_RECORD = []
rocket_input_vector_RECORD = []
time_array = []

# Simulate Physical System, keep track of variables
for t in range(1, 20000, 1):

  #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
  #                                                                                                                                                                            #
  #  This is Where the CNN and the MPC are called to make decisions                                                                                                            #
  #                                                                                                                                                                            #
  #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

  # Activate the CNN to recalculate optimal landing spot every 20 seconds
  if(t%200 == 1):
    new_coords = landing_coords(desired_state_vector[0], desired_state_vector[1], "/content/drive/MyDrive/Artificial Intelligence/Colab Files/ET Landing/render/render0003.png")

    desired_state_vector[0] = new_coords[0]
    desired_state_vector[1] = new_coords[1]

  # Call Model predictive Control

  reference_input = generate_reference(rocket_state_vector, desired_state_vector)

  rocket_input_vector = model_predictive_control(rocket_state_vector.copy(), rocket_input_vector.copy(), reference_input, g, d_t, t)

  #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
  #                                                                                                                                                                            #
  #  End of Rocket Landing Algorithm Interference                                                                                                                              #
  #                                                                                                                                                                            #
  #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

  print("EXECUTED = " + str(rocket_input_vector))

  # Simulate One Time Step
  simulate_physical_system(rocket_state_vector, rocket_attribute_vector, rocket_input_vector, g, d_t)

  cost_function(rocket_state_vector, desired_state_vector, True)

  # Record every variable in simulation
  rocket_state_vector_RECORD.append(rocket_state_vector.copy())
  desired_state_vector_RECORD.append(desired_state_vector.copy())
  rocket_attribute_vector_RECORD.append(rocket_attribute_vector.copy())
  rocket_input_vector_RECORD.append(rocket_input_vector.copy())
  time_array.append(t * d_t)

  # End if rocket touches ground
  if(rocket_state_vector[2] < 0):
    break

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#  Display Simulation Results                                                                                                                                                #
#                                                                                                                                                                            #
#  Displaying the results in the graphs allows for better interpritation                                                                                                     #
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Create the names of the graphs corresponding with the variables
rocket_state_vector_NAMES = ["X Coordinate (m)", "Y Coordinate (m)", "Z Coordinate (m)", "X Velocity (m/s)", "Y Velocity (m/s)", "Z Velocity (m/s)", "Theta S Orientation (radians)", "Theta D Orientation (radians)", "Theta S Angular Velocity (radians/s)", "Theta D Angular Velocity (radians/s)", "Dry Mass (kg)", "Wet Mass (kg)"]
desired_state_vector_NAMES = ["X Coordinate (m)", "Y Coordinate (m)", "Z Coordinate (m)", "X Velocity (m/s)", "Y Velocity (m/s)", "Z Velocity (m/s)", "Theta S Orientation (radians)", "Theta D Orientation (radians)", "Theta S Angular Velocity (radians/s)", "Theta D Angular Velocity (radians/s)", "Dry Mass (kg)", "Wet Mass (kg)"]
rocket_attribute_vector_NAMES = ["Max Engine Thrust", "Max Colg Gas Thruster Thrust", "Length from COG to EIP", "Length from COG to CGTIP", "Fuel Burn Rate"]
rocket_input_vector_NAMES = ["Engine Throttle", "Engine Theta S Gimbal", "Engine Theta D Gimbal", "CGT Theta S CW Throttle", "CGT Theta S CC Throttle", "CGT Theta D CW Throttle", "CGT Theta D CC Throttle"]

# Automate graph creating in the rocket_state_vector array, graph in red to signify rocke state vector
for i in range(0, len(rocket_state_vector), 1):
  plt.title(rocket_state_vector_NAMES[i] + " over Time")
  plt.xlabel("Time (seconds)")
  plt.ylabel(rocket_state_vector_NAMES[i])

  data_array = []

  for instance in rocket_state_vector_RECORD:
    data_array.append(instance[i])

  plt.plot(time_array, data_array, color = "red")
  plt.show()

# Automate graph creating in the desired_state_vector array, graph in orange to signify desired state vector
for i in range(0, len(desired_state_vector), 1):
  plt.title(desired_state_vector_NAMES[i] + " over Time")
  plt.xlabel("Time (seconds)")
  plt.ylabel(desired_state_vector_NAMES[i])

  data_array = []

  for instance in desired_state_vector_RECORD:
    data_array.append(instance[i])

  plt.plot(time_array, data_array, color = "orange")
  plt.show()

# Automate graph creating in the rocket_attribute_vector array, graph in green to signify rocke attribute vector
for i in range(0, len(rocket_attribute_vector), 1):
  plt.title(rocket_attribute_vector_NAMES[i] + " over Time")
  plt.xlabel("Time (seconds)")
  plt.ylabel(rocket_attribute_vector_NAMES[i])

  data_array = []

  for instance in rocket_attribute_vector_RECORD:
    data_array.append(instance[i])

  plt.plot(time_array, data_array, color = "green")
  plt.show()

# Automate graph creating in the rocket_input_vector array, graph in blue to signify rocket input vector
for i in range(0, len(rocket_input_vector), 1):
  plt.title(rocket_input_vector_NAMES[i] + " over Time")
  plt.xlabel("Time (seconds)")
  plt.ylabel(rocket_input_vector_NAMES[i])

  data_array = []

  for instance in rocket_input_vector_RECORD:
    data_array.append(instance[i])

  plt.plot(time_array, data_array, color = "blue")
  plt.show()

# Plot position of rocket x/z
plt.title("Position of Rocket")
plt.xlabel("x axis")
plt.ylabel("z axis")

x_values = []
z_values = []

for instance in rocket_state_vector_RECORD:
  x_values.append(instance[0])
  z_values.append(instance[2])

plt.scatter(x_values, z_values, color = "purple")
plt.show()

# Plot position of rocket x/y
plt.title("Position of Rocket")
plt.xlabel("x axis")
plt.ylabel("y axis")

x_values = []
y_values = []

for instance in rocket_state_vector_RECORD:
  x_values.append(instance[0])
  y_values.append(instance[1])

plt.scatter(x_values, y_values, color = "purple")
plt.show()

# Plot position of rocket y/z
plt.title("Position of Rocket")
plt.xlabel("y axis")
plt.ylabel("z axis")

y_values = []
z_values = []

for instance in rocket_state_vector_RECORD:
  y_values.append(instance[1])
  z_values.append(instance[2])

plt.scatter(y_values, z_values, color = "purple")
plt.show()

# 3D Trajectory Graph

x = []
y = []
z = []

for instance in rocket_state_vector_RECORD:
  x_values.append(instance[0])
  y_values.append(instance[1])
  z_values.append(instance[2])

ax = plt.axes(projection='3d')
ax.scatter3D(x_values, y_values, z_values, cmap='Greens');
