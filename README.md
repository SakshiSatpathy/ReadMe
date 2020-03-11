# CSProjects
Computer Science Projects' Description

This Github has private code from homework, labs and projects from UC Berkeley's CS61B: Data Structures class and for projects from Henry M. Gunn High School's Functional Object-Oriented Programming and AP Computer Science classes. For viewing this code, please send me requests to the specific private repositories so that I know you are not a student in the class. 

With respect to projects:

CS61B: Data Structures
Project 0: Signpost
Recreated the puzzle game Signpost, which is one of Simon Tatham's collection of GUI games. Given an incomplete Java Model-View-Controller program that creates these puzzles and allows its user to solve them, created a board in the Model class with all variables required to capture its state at any given time, used the Place class to access and modify the position of players, wrote methods to randomly generate new games in the Puzzle Generator class, and modified the Board Widget class to display the puzzle board. 

Project 1: Enigma
First large-scale project of CS61B, where I replicated the WWII German encryption machine "Enigma" by building a generalized simulator that could handle numerous different descriptions of possible initial configurations of the machine and messages to encode or decode. Worked mostly with Java's String, HashMap, ArrayList, and Scanner data structures to handle string manipulation, data mapping required, and file reading for encryption. 

Project 2: Tablut
Created a replica of the Swedish checkerboard game Tablut. The objective of the game is to capture the king before he reaches the edge of the board according to the specified legal moves the opposite side can make. A side loses when they have no legal moves, or if their move returns the board to a previous position. To read the detailed rules, please see the project spec here: http://inst.eecs.berkeley.edu/~cs61b/fa19/materials/proj/proj2/index.html. I built both the GUI version and the command line version of the game--including the board, moves, and implementing both manual players and AI players. For AI player, used game trees and alpha beta pruning based on heuristic values.

Project 3: Gitlet
Final project for the class. The project was to create a version control system (Gitlet) mimicking some of the basic functionality of the popular version-control system git from scratch. Functions implemented were add, commit, remove, log, status, checkout, merge, branch, and reset.
