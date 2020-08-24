# ReadMe
Computer Science Projects' Description

This Github has private code from homework, labs and projects from the following classes taken at UC Berkeley: Compsci C100 (Principles and Techniques of Data Science), CompSci 61B (Data Structures), CompSci 61A (Structure and Interpretation of Computer Programs). 

Selected Projects:

1) CompSci C100 (Principles and Techniques of Data Science)

Email Spam Filter: Built a filter after training data (reading, cleaning, feature engineering, modelling, fitting, split+testing) on logistic classifier. Bettered the model after extensive feature engineering through correlation plotting and visualization to eliminate multicollinearity, also using cross-validation and regularization during the multiple rounds of training to minimize bias and variance on unseen data. Achieved 99% training accuracy and 98% test accuracy.

Food Safety: Investigated San Francisco restaurant food safety scores using SQL and Pandas to clean and visualize data for valid zipcodes, examining lowest and highest scores’s violations over time, and progress in ratings over multiple inspections. Personally designed experiment to explore  potential causes of missing scores usually removed: found Complaints, New Ownership, and Reinspection/Followup to be leading correlators. 

2) CompSci 61B (Data Structures)

Gitlet: Designed and created version-control system with a subset of Git’s features, including the init,add, commit, rm, log, status, branch, rm-branch. Implemented serialization to store and read objects, used all data structure knowledge to initialize directories, generate  files, update branches and versions through objects and classes for each command. 

Enigma: Replicated the WWII German encryption machine "Enigma" by building a generalized simulator that could handle numerous different descriptions of possible initial configurations of the machine and messages to encode or decode. Worked mostly with Java's String, HashMap, ArrayList, and Scanner data structures to handle string manipulation, data mapping required, and file reading for encryption.

Tablut:  Built both the GUI version and the command line version of this chess-like game--including the board, moves, and implementing both manual players and AI players. For AI player, used game trees and alpha beta pruning based on heuristic values for generating optimal moves. 

Signpost: Recreated the puzzle game Signpost, which is one of Simon Tatham's collection of GUI games. Given an incomplete Java Model-View-Controller program that creates these puzzles and allows its user to solve them, created a board in the Model class with all variables required to capture its state at any given time, used the Place class to access and modify the position of players, wrote methods to randomly generate new games in the Puzzle Generator class, and modified the Board Widget class to display the puzzle board.


3) CompSci 61A (Structure and Interpretation of Computer Programs)

Scheme Interpreter: developed a Python interpreter for a subset of the Scheme language. After examining the design of our target language, built an interpreter with functions that read Scheme expressions, evaluate them, and display the results. Used all knowledge learned from CompSci 61a to apply to this final project. 

Ants: Replicated tower defense game Ants vs. Bees both GUI and command-line version. Used object-oriented programming to create, update gamestate, and move different ants and bee objects for ants to win the game with different classes, methods, attributes, objects, list comprehensions. Applied  abstraction, polymorphism, inheritance among other OOP concepts. 

CATS (CS61A AutoCorrected Typing Software): Wrote program measuring typing speed. Additionally, implemented a feature to correct spelling of a word after a user types it. Some helper functions extracted relevant text from selected paragraphs, and were swap functions to add distances of non-matching elements, and compute (minimum) edit distance. Used concepts of recursion, higher-order functions, self-reference, abstraction, and structures including nested loops, dictionaries and list-comprehensions. 

Hog: Developed a simulator and multiple strategies for the two-player dice game Hog. Helper functions implemented included applications of special rules like pig out, free bacon, feral hog, swine swap, and functions to simulate taking turns till the fastest player gets to the max score of 100 based on helper functions for score-maximizing strategies (highest average turn score). Commentary functions implemented also announced the players’ score after each turn, the lead score changes when applicable, and when a certain player’s score increases the highest during a specific game. Used higher-order functions and control statements, along with various applications of print, lambda, function calls and casting. 


For viewing this code, please send me requests at sakshi.satpathy@berkeley.edu so that I know you are not a student in the class. 
