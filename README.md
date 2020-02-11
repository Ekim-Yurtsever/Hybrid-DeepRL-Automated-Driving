# A Hybrid Deep Reinforcement Learning Based Automated Driving Agent for CARLA

Codebase for our Hybrid Deep Reinforcement Learning (H-DRL) based automated driving project.
The related paper can be accessed with [this](https://arxiv.org/pdf/2002.00434.pdf) link.

## Work in progress. Stay tuned for the full release.

## Features

## Overview

<img src="gifs/overview.png" title="The proposed method"> 

An overview of our framework. The proposed system is a hybrid of a model-based planner and a model-free DRL agent. *Other sensor inputs can be anything the conventional pipe needs. ** We integrate model-based planners into the DRL agent by adding "distance to the closest waypoint" to our state-space, where the path planner gives the closest waypoint. Any kind of path planner can be integrated into the DRL agent with the proposed method.

## Installation

## Credits
This project was forked from a conventional DRL implementation for CARLA by Sentdex. https://github.com/Sentdex/Carla-RL
