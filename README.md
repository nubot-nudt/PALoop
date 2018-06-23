# Pose-Appearance-based Loop Closure Detection for Non-Linear Optimization-based SLAM

## Description

The loop closure detection is a key issue to eliminate the accumulative error of visual odometry. In this work, a novel pose-appearance-based loop closure detection method is proposed, inspired by the complementarity of the pose and the appearance.
The pose-based loop closure probability model is derived based on the non-linear optimization model of the visual odometry. Then the combination of the pose-based probability and any existing appearance-based loop closure detection method is detailed. 

This project cantians the basic implementation of the pose-based loop closure probability, in the files of ProbabilisticLoopClosing.h and ProbabilisticLoopClosing.cpp. 

The proposed loop closure detection method can ONLY work together with a on-linear optimization-based visual odometry. The demonstration of applying this method in ORB-SLAM2 is provided in this project.

Maintainer: NuBot workshop, NUDT China - http://nubot.trustie.net and https://github.com/nubot-nudt

