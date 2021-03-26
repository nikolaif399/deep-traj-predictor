import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
#sys.path.append(os.path.abspath('../libs'))

def createWall(p, pos, orn, width, height, cor=1):
  wallCollId = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                           halfExtents=[width/2,height/2,0.01])
  wallVisualId = p.createVisualShape(shapeType=p.GEOM_BOX,
                                     halfExtents=[width/2,height/2,0.01])

  wallId = p.createMultiBody(baseMass=0,
                    baseCollisionShapeIndex=wallCollId,
                    baseVisualShapeIndex=wallVisualId,
                    basePosition=pos,
                    baseOrientation = p.getQuaternionFromEuler(orn))

  p.changeDynamics(wallId,-1, restitution=0.99)
  return wallId

def runSim(gui=True, record = False, N=1000, filename="data.npz", lookaheadSteps=50):
  # Attach sim to physics client
  if gui:
    physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
  else:
    physicsClient = p.connect(p.DIRECT) 

  p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

  # Update simulation world
  dt = 0.01
  p.setTimeStep(dt)
  p.setGravity(0,0,-9.81)
  planeId = p.loadURDF("plane.urdf")
  p.changeDynamics(planeId,-1,
                   restitution=0.99)

  # Load walls
  wallLeft = createWall(p, [1,0,1], [0,np.pi/2,0], 2,2)
  wallBack = createWall(p, [0,-1,1], [0,np.pi/2,np.pi/2], 2,2)
  wallRight = createWall(p, [-1,0,1], [0,np.pi/2,0], 2,2)

  # Load ball
  ballRadius = 0.1
  ballCollisionId = p.createCollisionShape(shapeType=p.GEOM_SPHERE,
                                           radius=ballRadius)
  ballVisualId = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                           radius=ballRadius,
                                           rgbaColor=[1,0,0,1])
  ballId = p.createMultiBody(baseMass=1,
                    baseInertialFramePosition=[0, 0, 0],
                    baseCollisionShapeIndex=ballCollisionId,
                    baseVisualShapeIndex=ballVisualId,
                    basePosition=[0,0,2])

  p.changeDynamics(ballId,-1, restitution=0.99)
  p.resetBaseVelocity(ballId,linearVelocity=[-5,0,0])

  # Change camera pose
  p.resetDebugVisualizerCamera(3.2,-184,-40,[0,0,0])
  
  # Start logging video
  if (record):
    logger = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,"ball_bouncing.mp4")

  # Collect all states and sensor readings
  states = np.empty((N,7))

  # Run through trials
  time.sleep(1)
  
  for i in range (N):
    # Ray traced sensors for nearest object

    ballPos = np.array(p.getBasePositionAndOrientation(ballId)[0])
    ballVel = np.array(p.getBaseVelocity(ballId)[0])

    ballLookaheadPos = ballPos + lookaheadSteps*dt*ballVel
    
    ray = p.rayTest(ballPos,ballLookaheadPos)

    rayFraction = 1 # Default value for ray fraction
    rayFraction = ray[0][2]
    p.addUserDebugLine(lineFromXYZ=ballPos,
                       lineToXYZ=ballPos + (ballLookaheadPos-ballPos)*rayFraction,
                       lineColorRGB=(0,1,0),
                       lifeTime=2*dt)

    states[i,0:3] = ballPos
    states[i,3:6] = ballVel
    states[i,6] = rayFraction

    # Update our dynamics
    p.stepSimulation()
    if gui:
      time.sleep(dt)
      
  if (record):
    p.stopStateLogging(logger)

  np.savez_compressed(filename,data=states)

  p.disconnect()

if __name__ == '__main__':
  runSim(gui=True, record=False, filename="data.npz")