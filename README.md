# (A) Computer Science Projects' Description 

Note: As of June 2023, there are 25 (20 private, 5 public) repositories with coding projects

This Github (SakshiSatpathy) has private code from homework, labs and projects from the following CS (Computer Science), EECS (Electrical Engineering and Computer Science), and DS (Data Science) classes I have taken at UC Berkeley: 
1) CompSci 280 (Graduate Computer Vision)
2) CompSci 182 (Designing, Visualizing and Understanding Deep Neural Networks)
4) CompSci C100 (Principles and Techniques of Data Science) 
5) CompSci 189 (Introduction to Machine Learning)
6) EECS 127 (Optimization Models in Engineering)
7) DATA C102 (Data, Inference, and Decisions)
8) DATA 101 (Data Engineering) 
9) EECS 126 (Probability and Random Processes)
10) CompSci 161 (Computer Security)
11) DES-INV 190-10 (Design and Cybersecurity) 
12) CompSci 170 (Efficient Algorithms and Intractable Problems)
13) INFO 159 (Natural Language Processing)
14) CompSci 160 (User Interface Design and Development)
15) CompSci 61B (Data Structures) 
16) CompSci 61A (Structure and Interpretation of Computer Programs) 


Selected Projects:

## 1) CompSci 280 (Graduate Computer Vision)
**Final Project:** Facial Landmark Detection detects and localizes points such as the eyes, nose, mouth, and chin. Designed project and applied different combinations of architectures and frameworks—specifically, different U-nets, knowledge distillation and denoting diffusion—to improve the performance of facial landmark detection using deep learning. 

**Assignment 1:** Constructed a lighter and smaller Residual Network (ResNet) architecture. Train different versions of network to perform image classification by minimizing the cross-entropy between the network’s prediction and CIFAR-10 targets. Used Adam optimizer. Plotted the loss value and the classification accuracy on training and validation set, and chose the iteration with the least error. 

**Assignment 2:** Created model for multi-view reconstruction that recovers the 3D shape of an object given its image from two different viewpoints. In particular, given 2D point correspondences in the 2 images, estimated their 3D position in the scene and also estimate the camera positions and orientations. Implemented camera model, fundamental matrix, essential matrix, eight-point algorithm, estimated extrinsic camera Parameters from the essential matrix, and triangulation. 

-----

## 2) CompSci 182 (Designing, Visualizing and Understanding Deep Neural Networks)

**Final Project:**
The paper *High-Resolution Image Synthesis with Latent Diffusion Models* introduces a way to train and run diffusion models at a much lower compute cost while still maintaining the sample quality. The paper’s approach can be broken down into 2 subtasks: training an autoencoder for dimensionality reduction and training a denoising UNet for the diffusion process on the latent space. Our group designed a problem set and corresponding solutions to guide students through both of the subtasks through 5 main questions. The first 2 questions encourage the students to read the paper and think about the pain points of traditional diffusion models, how latent diffusion models address them, and understand details on how the model is set up. The 3rd question guides the students through the important mathematical part of the diffusion model so they would be able to implement a simplified version of their own later on. The 4th question lets the students code their own simplified diffusion model on a simple dataset that is easy to visualize so they would understand all the components in training their own diffusion model. More details on each of the questions are listed in the commentary.pdf file in the project repository. The 5th question lets the students implement and train their own autoencoder (variational autoencoder) in the style of the paper, making them understand the different autoencoder loss terms and how each of them affects the reconstruction quality.

**HW12:**

**Reinforcement Learning from Human Feedback:** Implemented the RLHF algorithm to solve a NLP task  (news summarization). First, generated multiple hypotheses for each input example using a small unlabeled training set. Second, obtained human feedback by having "humans" rank these hypotheses according to their quality. Third, trained a reward model that captures human preferences by fitting it to the collected human feedback. Fourth, leveraged the reward model to enhance our summarization model using reinforcement learning techniques.
**Early Exit:** Implemented baseline ResNet and ResNet with Early Exit (Global average pooling, MLP layer of 128 nodes, ReLU, MLP layer of num_class nodes). Compared performance and outputs of early exit result and the final result. Did joint training (Training the entire network with all the exits at the same time). 

**HW11:**

**Policy Gradient in Reinforcement Learning:** Policy gradient at a high level approximates the gradient and then does gradient descent using this approximated gradient. Implemented policy gradient algorithm for neural networks, and compared accuracy with baseline supervised learning approach. 

**Transformer for Summarization (Part II):** Efficiently enabled the Transformer encoder-decoder model to generate sequences. Then fine-tuned another Transformer encoder-decoder model based on the pretrained language model T5 for the same task. Finally, compared the performance of the fine-tuned model against our previous custom model that was trained from scratch.

**Generative Models:** Used PyTorch to implement the variational autoencoder (VAE) and learn a probabilistic model of the MNIST dataset of handwritten digits. Also trained a Generative Adversarial Network (GAN) on MNIST dataset.

**HW10:**

**MAML:** Implemented Meta-Learning on 1-D Functions for both regression and classification. Also compared performance of the meta-learned feature weights with the case where all feature weights are 1, and with the case where the oracle is used (which performs logistic regression using only the features present in the data). 

**Pruning:** Using the CIFAR10 dataset and VGG network, implemented and applied fine-graining pruning, implemented and applied channel pruning, and compared baseline net with pruned network, and compared performance of fine-grained vs channel pruning. 

**Quantization:** Quantized a classical neural network model to reduce both model size and latency. Specifically, implement and applied k-means quantization, quantization-aware training for k-means quantization, linear quantization, and integer-only inference for linear quantization. Then compared baseline net with quantized net performance. Also compared performance and tradeoffs between the above quantization approaches. 

**HW9:**

**MAE:** Implemented Vision Transformer (ViT) and trained it on CIFAR dataset. Also, implemented Masked Autoencoder (MAE).

**Transformer for Summarization (Part I):** Implemented a Transformer using fundamental building blocks in PyTorch. Then applied the Transformer encoder-decoder model to a sequence-to-sequence NLP task: document summarization.

**HW8:**

**Implemented transformer model:** Created a hand-designed transformer model capable of solving a basic problem. Analyzed the attention patterns of a trained network to gain insights into how learned models often utilize features that differ greatly from those employed by humans.

**HW7:**

**Autoencoders:** Implemented pretraining models with unsupervised learning and evaluating the learned representations with a linear classifier. Specifically, we will implemented three different architectures: Vanilla Autoencoder, Denoising Autoencoder, and Masked Autoencoder. 

**HW6:**

**LSTMs:** Implemented simple RNNs and LSTMs (Long short-term memory neural networks with feedback connections), then explored how gradients flow through these different networks.

**RNN for Last Name Classification:** Trained a neural network to predict the probable language of origin for a given last name / family name in Latin alphabets. Firstly, preprocessed raw text data for suitable input into an RNN and LSTM. Then utilized PyTorch to train the recurrent neural network models. Finally, evaluated models’ performance and made predictions on unseen data.

**HW5:**

**Graph Clustering:** Implemented k-means clustering algorithm. Also interpreted a dataset as a graph (interpreted every single point in the provided dataset as a node in a graph. The implemented adjacency matrix, stochastic matrix and SVD decomposition to relate every node in the graph is such way that they points that are closer together maintain that relationship while points that are farther are explicitly identified. Then performed k-means clustering on graph re-interpretation to improve performance of clustering for 3 classes. 

**Zachary's Karate Club:** Trained a GNN (Graph Neural Network), specifically a multi-layer Graph Convolutional Network, to cluster people in the karate club in such that people who are more likely to associate with either the officer or Mr. Hi will be close together, while the distance beween the 2 classes will be far. 

**HW4:**

**Dropout:** Explored the effect of dropout on a simple gradient descent problem. Specifically trained weights to solve linear equation where weights initialized to be 0. Formulated question as OLS optimization problem. Compared performance of the following layers on network (trained on CIFAR-10 dataset): *No dropout + Least-Squares* vs. *No dropout + Gradient Descent* vs. *Dropout + Least-Squares* vs. *Dropout + Gradient Descent*

**Edge Detection:** Implemented and compared convolutional neural networks (CNNs) and multi-layer perceptrons (MLPs) to understand what is inductive bias and how it affects the learning process, and to conduct systematic ML experiments. Used the edge detection task to study the inductive bias of CNN. Specifically, generated synthetic dataset, and made models overfit this small dataset to test model architecture. 

**HW3:** 

**Hand-Designing Filters:** Designed convolution filters by hand to understand the operation of convolution. Specifically, designed simple blurring (Averaging filter) and edge detection (Laplacian derivative) filters. 

**CNN with PyTorch:** Practiced writing backpropagation code and training Neural Networks. Specifically, implemented spatial batch normalization (forward and backward pass), different layer types for 3-layer convolutional network (convolutional forward pass, convolutional backward pass, max pooling). Checked performance with gradient check and implementing loss functions. 

**HW2:**

**Fully Connected Net with PyTorch from HW1:** Implemented Two Layer  fully-connected neural network with ReLU nonlinearity and softmax loss (affine - relu - affine - softmax). Instead of gradient descent, interacts with Solver object that runs optimization (see below). 

**Optimization and Initialization:** Implemented different optimization Methods and initialization to speed up learning and get a better final value for the cost function (as opposed to gradient descent). Specifically, implemented Stochastic Gradient Descent (SGD), SGD with Momentum, and compared performance with regular gradient descent. 

**Accelerating Gradient Descent with Momentum:** Implemented gradient descent with momentum with different learning rates and compared and visualized performance (gradient changes and loss changes with different iterations)

**Linearization (Part 2):** Trained a 1-hidden layer neural network using Stochastic Gradient Descent with Momentum. Visualized this model using local linearization (first-order Taylor expansion) and visualized decomposition of principle components using SVD decomposition. 


**HW1:**

**Linearization (Part 1):** Learned the piecewise linear target function using a simple 1-hidden layer neural network with ReLU non-linearity. Also created two SGD optimizers to choose whether to train all parameters or only the linear output layer's parameters. Implemented several versions of this network with varying widths to explore how hidden layer width impacts learning performance.

**Fully Connected Net with PyTorch:** Implemented two-layer fully-connected network using a modular approach. For each layer (affine, ReLU non-linearity), implemented a forward and a backward function. Additionally, implemented Softmax and SVM loss functions. Final architecture was affine - relu - affine - softmax, and uses gradient descent for optimization.

----------------------------------------
## 2) CompSci C100 (Principles and Techniques of Data Science)

**(a) Email Spam Filter:** Built a filter after training data (reading, cleaning, feature engineering, modelling, fitting, split+testing) on logistic classifier. Bettered the model after extensive feature engineering through correlation plotting and visualization to eliminate multicollinearity, also using cross-validation and regularization during the multiple rounds of training to minimize bias and variance on unseen data. Achieved 99% training accuracy and 98% test accuracy.

**(b) Food Safety:** Investigated San Francisco restaurant food safety scores using SQL and Pandas to clean and visualize data for valid zipcodes, examining lowest and highest scores’s violations over time, and progress in ratings over multiple inspections. Personally designed experiment to explore  potential causes of missing scores usually removed: found Complaints, New Ownership, and Reinspection/Followup to be leading correlators. 

---

## 3) CompSci 189 (Introduction to Machine Learning) Homeworks:

**Seven homeworks on the following Machine Learning topics:**

(a) Maximum Likelihood Estimation (MLE), Maximum a posteriori (MAP) estimation

(b) Multivariate Gaussians, Classification for Gaussians, Linear Regression

(c) Logistic Regression, Gaussian Discriminative Analysis, Support Vector Machines (SVMs), 

(d) Kernels, Validation Sets, Bias-Variance and Decision Theory, Precision, Recall and ROC Curves, 

(e) K-Nearest Neighbors, Decision Trees, 

(f) Principal Component Analysis (PCA), Clustering, Neural Networks

(g) Convolutional Neural Networks, Transformers, Unsupervised Learning Methods, and Recommender Systems. 

---

## 4) EECS 127 (Optimization Models in Engineering) HomeWork

**Problem sets/homework on the following topics:**

(a) Linear Algebra concepts (including vectors, projections, matrices, symmetric matrices, Linear equations, least-squares and minimum-norm problems, SVD, PCA and related optimization problems)

(b) Convex Optimization problems on convex sets, convex functions, KKT optimality conditions, duality (weak and strong), Slater's condition, special convex models (including LP, QP, GP, SOCP), and Robustness

(c) Optimization applications including Machine Learning, Control Systems, Engineering design, and Finance

---

## 3) DATA C102 (Data, Inference and Decisions) Final Project and Classwork

**Final Project:**

(a) Topic 1: GLMs and Non-parametric Methods: Predicted mortality counts from asthma given location, gender, and race using both general linear models and non-parametric models using the CDC: Annual State-Level U.S. Chronic Disease Indicators [Filtered for Asthma] dataset. Built random forests and decision trees using bootstrap aggregation. For the GLM, used a Negative Binomial regression distribution family. 

(b) Topic 2: Bayesian Hierarchical Modeling: Predicted hospitalizations for cardiovascular diseases based on population gender and race demographics, holding age constant, using a Beta-Binomial mixture model after drawing from the CDC: Annual State-Level U.S. Chronic Disease Indicators (Filtered for Cardiovascular Disease) dataset. 

**Ten labs tackling real-world challenges such as:**

**(a) Lab 1:**  
Part 1: Implemented concepts of decision theory, including testing, p-values, and controlling False Discovery Rate (FDR). Specifically, I wrote functions to calculate the likelihood ratio, calculate the probability of false positives, create an alpha level decision rule the rejects the null hypothesis at level alpha, compute p-values (smallest alpha for which test rejects null), and predicting whether samples are from the null or alternative distribution based on its p-value. 

Part 2: I then explored controlling for the probability of false discoveries for multiple hypothesis testing. Specifically, I implemented:  
1. Naive thresholding (ignoring that multiple testing is happening)
2. Bonferroni correction that corrects p-values by controlling the Family Wise error rate (to account for multiple testing)
3. Benjamini-Hochberg procedure for multiple hypothesis testing
I also created confusion matrices to report the results on true positives, true negatives, false positives, and false negatives. 

**(b) Lab 2:** I examined a medical diagnosis case study for hypothesis testing and utilized loss functions to optimize decisions. 
Part 1: Given test kit data, I implemented functions that computed average empirical loss and computed the average loss (empirical risk) with respect to various levels of 𝛼. I then investigated the average loss plot for different levels of disease prevalence. 
Part 2: Given test kit data, I implemented functions that computed the posterior probability that the patient truly has the disease conditioned on a positive test result, computed the expected loss function with respect to the posterior distribution, and decided whether or not to administer the treatment by comparing the expected losses in each case. 

**(c) Lab 3:** I attempted to estimate the COVID infection risk in households by curating multiple studies to get the best estimate of the Secondary Attack Rate (SAR), and find regions with the lowest and highest SAR. Specifically, I implemented functions that computed the trivial estimate of SAR, Examined the prior distribution, and computed the posterior mean minimizes the Bayes Risk for the Squared Error Loss of a Beta-Binomial model, and approximated inference using a PyMC3 Beta-Binomial model. 

**(d) Lab 4:** I partially implemented three sampling strategies for obtaining samples from unknown distributions, specifically, rejection Sampling (sampling from 1D and 2D density functions), Gibbs Sampling (building a Gibbs sampler), and Metropolis Hastings (interpreting all results for varying variance levels)

**(e) Lab 6:** I explored and interpreted several nonparametric methods for regression to predict the price of hybrid cars using other features of the cars.   
Part 1 and 2: Specifically, I implemented functions that split the hybrid car data into train and test sets, and predicted the output by building, fitting and predicting from several models: including linear regression, decision trees for regression, and random forests. I then compared the performance of each of the models.  
Part 3: I then explored the effect of feature engineering on the interpretability of a given model using a toy dataset. Specifically, I implemented functions that added random features to the dataset based on a sigmoid transformation of a linear combination of the data. 

**(f) Lab 7:** I estimated the causal effect of the number of books and income on the SAT Score given that the number of books is observed while the family income is unobserved. Specifically, I implemented code that first did the above using Ordinary Least Squares. Secondly, I used 2-stage least squares and instrumental variables to eliminate the bias from the unobserved variable family income. The first stage "predicts" the number of books a student read from whether or not they had a readathon. The second stage regresses the SAT score onto the predicted number of books read. 

**(g) Lab 8:** I explored the challenges of doing causal inference without randomization using the unconfoundedness assumption, and partially reproduced results from a real labor economics application in a very famous paper by Robert Lalonde that estimates the causal effect of a training program from the 1970s on income. Specifically, I implemented code that computed the causal effect in randomized experiments, computed the Simple Difference in Observed group means (SDO) by not using a control group (for observational study data). 

**(h) Lab 9:** I implemented and gained a better understanding of the Multi-armed bandits problem through this lab. Specifically, the pros and cons of the Upper Confidence Bounds (UCB) and Thompson Sampling algorithms for the multi-armed bandits problem. The first algorithm I implemented pulled the choice of arm given the frequentist take on multi-armed bandits, otherwise known as the Upper Confidence Bounds (UCB) algorithm. I then investigated the pseudo-regret of the UCB algorithm. The second algorithm I implemented pulled the choice of arm given the Bayesian take on multi-armed bandits, known as Thompson Sampling. In this setting, we begin with a prior over the mean of each arm. Finally, I evaluated the pros and Cons of UCB and Thompson Sampling given their pseudo-regret and implementation. 

**(i) Lab 10:** I explored solving MDPs (Markov Decision Processes) by collecting data. This included simple Monte Carlo estimates from offline data, and the online Q-learning algorithm.   
Part 1: Specifically, I wrote some code which takes a policy (in the form of a function from states to actions) and then runs that policy in a given GridWorld environment so that we can collect a dataset. Next, I computed the discounted sum of rewards.   
Part 2: I also implemented code that retrieved the optimal policy from the optimal Q-function, and other functions which updated the Q function using observed samples when the optimal Q-function is unknown, and creates agents that can be run in  deterministic or stochastic settings. 

**Five labs tackling real-world challenges such as:  **

**(a) HW1:** Coding Problems solved included:
1. Math Stats: ranging from conditional probabilities to computing row-wise and column-wise rates to computing the likelihood ratio
2. Bias in Police Stops:  Coding a normalized histogram for the theoretical null given police dataset z-scores, computing p-values and applying the BH procedure to find the number of discoveries for the empirical and theoretical null distributions
3. p-values, FDR and FWER: Coding a function avg_difference_in_means  compute the p-value for given null and alternative distributions for each feature subselection, and determining  for which tests do we reject the null hypothesis in case of controlling for the FDR vs. the FWER given thresholds.  

**(b) HW2:** Coding Problems solved included:
1. The One with all the Beetles:  To estimate the size of the largest possible beetle using statistical modeling, found the likelihood function and MLE, computed the posterior and showing it is Pareto-distributed given a Pareto prior, and interpreting the alpha and beta parameters. Additionally, wrote code to generate the posterior, and used the data to make a graph of one curve for each of the days 1, 10, 50 and 100 (so four curves total), where each curve is the PDF of the posterior for the respective day
2. Bayesian Fidget Spinners: Created a Beta-Bernoulli-Geometric model (similar to Gaussian mixture model) to classify which boxes come from factories, and how reliable each factory is based on their fidget spinner production
3. Rejection Sampling
 
**(c) HW3:** Coding Problems solved included:
1. GLM for Dilution Assay: Reformulated problem as generalized linear models to estimate the unknown concentration ρ0 of an infectious microbe in a solution given dilution
2. Image Denoising with Gibbs Sampling: Derived a Gibbs sampling algorithm to restore a corrupted image
3. Bayesian GLM: Applied Gaussian linear regression to election data to predict the outcome of the 2020 election using information from previous elections

**(d) HW4:** Coding Problems solved included:
1. Observational Data on Infant Health: Estimated the causal effect of The Infant Health and Development Program on the child’s cognitive test scores using logistic regression and devising a propensity score model to control for observed confounders in the observational study data
2. Causal Inference Potpourri: Evaluated study design effectiveness of a new veterinary drug for sick seals using graphical modeling and the backdoor criterion. 

**(e) HW5:** Coding involved the Simulation Study of Bandit Algorithms:
1. Coding the explore-then-commit algorithm and computing pseudo-regret. 
2. Coding the UCB algorithm and computing pseudo-regret: 
3. Comparing the distributions of the rewards by also plotting them on the same plot and briefly justify the salient differences 

## 4) DATA 101 (Data Engineering) Projects
**(a) Project 1: SQL**
Worked with SQL on the IMDB database to explore and extract relevant information from database with SQL functions, perform data cleaning and transformation using string functions and regex, and use the cleaned data to run insightful analysis using joins, aggregations, and window functions

**(b) Project 2: Query Performance**
Worked with the Lahman's Baseball Database to explore how the database system optimizes query execution (and how users can further tune the performance of their queries using index selection, properties of query processing, and query optimization. 

**(c) Project 3: Data Transformation**
Worked with one month of unstructured sensor data from UC Berkeley buildings. Used data prep, data cleaning, normalization, entity resolution, linear interpolation, and outlier handling to transform the data to make it usable for data pipelines. 

**(d) Project 4: Mongo**
Investigated how different database systems handle semi-structured JSON data using MongoDB, Postgres SQL and Pandas. Worked with the Yelp Academic Dataset to understand what Mongo can (and cannot) do with regards to its documents as a NoSQL datastore and compare and contrast this to other data representation formats such as the relational model.


## 5) EECS 126 (Probability and Random Processes) Labs

**Nine labs tackling real-world challenges such as:**

(a) encoding and decoding messages

(b) simulating a “Guess the word” game with binary search and Huffman coding

(c) applying Central Limit theorem to entropy and confidence intervals calculations 

(d) developing a Metropolis-Hastings algorithm to simulate a discrete-time and continuous-time Markov Chains (these are just a few of the several scenarios). 

**The topics encompassed were:**

(a) Probability fundamentals (Discrete and Continuous Probability, Bounds, Convergence of Random Variables, Law of Large Numbers, Discrete Time Markov Chains) 

(b) Random processes and estimation (Transforms, Central Limit Theorem, Queueing, Poisson Processes, Continuous Time Markov Chains, Communication, Information Theory, MLE/MAP, Detection, Hypothesis Testing) 

(c) Applications of probability (LLSE, MMSE, Kalman Filtering, Tracking) 


## 6) CompSci 161 (Computer Security)
**HW1:** Security Principles, C Memory Review, C Stack Layout, GDB, Buffer Overflow Intro

**HW2:** Memory Safety Vulnerabilities (how to defeat stack canaries to exploit a program; identifying and preventing format string vulnerabilities), Block Ciphers, Padding, CBC Review, Padding Oracles

**HW3:** Cryptography, Hashing Functions, El Gamal Encryption, Length Extension, Padding Oracle Lab
CBC Padding Oracle Attack Lab: decrypted a message encrypted with AES-CBC using a padding oracle attack, 

**HW4:** Finding Common Patients using Symmetric-Key Encryption and Hashing, PRNGs, Diffie-Hellman Walkthrough, Signatures, Certificates, Passwords

**HW5:** SQL Injection, Web Security True/False, Snapitterbook, Go Tutorial, Project 2 Warm-Up (Threat Model, Design Requirements for Sharing, Receiving and Revoking), SQL Injection Lab
SQL Injection Lab: Implemented a SQL Injection Attack 

**HW6:** Network Security Intro (ARP, ARP Attack, DHCP, DHCP Attacks, BGP & IP, Defenses), Transport Layer (UDP, TCP Walkthrough, On-path TCP Hijacking, Off Path Attacker, Defenses), WPA on-path attacker, WPA in-path attacker, TLS

**HW7:** DNA Walkthrough, DNS Amplification Attack, Intrusion Detection Systems, Protecting REGULUS (Default Design, Stateless Packet Filtering (Default-Allow, Default-Deny), Stateful Packet Filtering (with Redirection, Public VPNs, Private VPNs), Firewall Robustness, Malware, Tor Basics

**Project 1 (Exploiting Memory Vulnerabilities):** Exploited a series of vulnerable programs on a virtual machine. This means that, if you provide a specially crafted input to the orbit program, you can cause it to execute your own, malicious code, called shellcode. We will write our input using Python 3, stored in an egg file. Whatever bytes are printed from the egg file will be sent as input to the vulnerable program. Note that at the top of all of our files, including the egg file. The shebang line tells the operating system that this executable should be run as a Python file:

**Project 2 (An End-to-End Encrypted File Sharing System):**
Applied the cryptographic primitives introduced in class to design and implement the client application for a secure file sharing system. File sharing system was similar to Dropbox, but secured with cryptography so that the server cannot view or tamper with your data. The client will be written in Golang and will allow users to take the following actions: authenticate with a username and password; save files to the server; load saved files from the server; overwrite saved files on the server; append to saved files on the server; share saved files with other users; and revoke access to previously shared files. Implementation used public-key encryption, digital signatures, Hash-Based Message Authentication Code (HMAC), Hash-Based Key Derivation Function (HashKDF), Password-Based Key Derivation Function (PBKDF), Symmetric Encryption, and Random-Byte Generators. 


**Project 3 (Breaching a Vulnerable Web Server):**
Exploited a poorly designed website by finding 8 vulnerabilities in the UnicornBox servers and implementing corresponding exploits. These included XSS attacks, CSRF attacks, and SQL Injection attacks. 


## 7) DES-INV 190-10 (Design and Cybersecurity) Final Project

**Cyberus:** Designed a one-stop-shop app for smart city stakeholders (residents, government, businesses) to understand and safeguard themselves using existing cybersecurity and privacy protections. 

Users can:

(a) find if they live in a smart city, load city-specific features including options to scan public tech and understand data collected from residents

(b) schedule and join city-wide events on the calendar (eg. smart city workers tackling interoperability, expert talks, or smart city government leadership training)

(c) Teachers can download the city-curated school curriculum on smart cities and how students can protect their privacy

(d) learn about government policies, local vendors, smart city definitions, and cyber firms near users

(e) buy custom-designed surveillance camera covers which are sensor-enabled and close surveillance cameras in the event of a ransomware or city-wide cyber breach. This prevents surveillance misuse by unauthorized parties.  

## 8) CompSci 170 (Efficient Algorithms and Intractable Problems) Project and Classwork
**Final Project:**
Built solver which attempted NP-Hard problem for placing placing penguins in igloos based efficiently based on exponential constraints. Algorithm features relaxed dynamic programming algorithm with knapsack-based approach that accounts for maximum profit at time t, and simulated annealing is applied to the DP solver's generated outputs to improve profit through randomly exploring permutations. 

**Problem sets/homework and discussions on the following algorithms:**

Graphs and Trees: Minimum spanning trees, depth-first and breadth-first search, topological sort with strongly-connected components, shortest paths algorithms (Dijkstra and Bellman-Ford), randomized min-cut

General approaches: Fast Fourier Transform, divide-and-conquer, dynamic programming, greedy algorithms, union find

Optimization: linear programming, duality, network flow

Theory: NP completeness, Approximation algorithms

Security: Hashing, Huffman Codes

## 9) INFO 159 (Natural Language Processing) Final Project
Designed and implemented a new document classification task Annotation Project to understand if, when given restaurant reviews from the Yelp dataset, one can discern the level of dining (fast food dining, casual dining, fine dining) based on the word and phrase descriptors within the review text bodies, and internal features during the machine learning process. Designed annotation task subject study, developed comprehensive guidelines for consistent third-party annotations, validated label-specific patterns through measuring the inter-annotator agreement rate and external feedback, and built several classifiers (logistic regression, BERT), comparing different models (eg. BERT, Bag of Words with L2 regularization, TD-IDF with L2 regularization, combined word-phrase features) and conducting a robust error analysis and report for future improvements to improve accuracy and class imbalance captured through the classifier performance based on the developed annotation guidelines.   
 
## 10) CompSci 160 (User Interface Design and Development) Projects

**(a) Auxilium:** Designed and built a community helping web app Auxilium which matches older people needing help and college students wanting to help. Supports chat, location sharing, and posting of blogs and tasks with a customizable home feed for users based on interest communities, skills, and location. Built using HTML, CSS, Django, Google Maps Location API, Websockets. 

**(b) Repartee:** Designed and built reading app to help busy book-lovers schedule time to read free books, while interacting with others who share their same passion for reading. Repartee allows users to create communities for books and upload their favorite titles to the online library to share with other Reparteers for free on the Desktop app. Users can then search for and access their preferred titles by book and community on the Phone app, and set an in-app timer in minutes to read these books while chatting with other readers in the in-app chat room. Built using HTML, CSS, Django, Websockets. 

**(c) Hue-man Healing:** Designed and built interactive Coloring App for iPhone X users. This app is designed for art therapy and de-stressing for busy working professionals with very small breaks in the day (5-30 minutes). Users can choose their coloring tool, and draw from scratch or access templates in varying themes to color in. While enjoying the app, they can set their notifications to “Do-not-disturb”, and can share their artifacts at the end. Built using HTML, CSS, Django. 

**(d) Project 2:** Built misbehaving bubble wrap, bubble wrap on slow internet, drag and drop boxes, responsive grid layouts, and flexbox css. These tasks used different aspects of HTML, CSS, and JQuery. 

**(e) Leisure Traveler:** Weather-based trip recommendation web app. Users set temperature preferences, so cities are categorized LAZY SAFE, UNSAFE, or STORM. They can then search for a city to view its temperature, precipitation, wind, and smog level over time, add the city to the Comparison List, Favorite it, or read travelers’ reviews. Users can also compare cities’ forecasts, and navigate using the GPS. Users can check the weather mid-trip, and switch location if the weather is LAZY UNSAFE. In case of disaster, users get alerted, and can find the city’s attractions, post on social media, and upload reviews.

## 11) CompSci 61B (Data Structures) Projects

**(a) Gitlet:** Designed and created version-control system with a subset of Git’s features, including the init,add, commit, rm, log, status, branch, rm-branch. Implemented serialization to store and read objects, used all data structure knowledge to initialize directories, generate  files, update branches and versions through objects and classes for each command. 

**(b) Enigma:** Replicated the WWII German encryption machine "Enigma" by building a generalized simulator that could handle numerous different descriptions of possible initial configurations of the machine and messages to encode or decode. Worked mostly with Java's String, HashMap, ArrayList, and Scanner data structures to handle string manipulation, data mapping required, and file reading for encryption.

**(c) Tablut:**  Built both the GUI version and the command line version of this chess-like game--including the board, moves, and implementing both manual players and AI players. For AI player, used game trees and alpha beta pruning based on heuristic values for generating optimal moves. 

**(d) Signpost:** Recreated the puzzle game Signpost, which is one of Simon Tatham's collection of GUI games. Given an incomplete Java Model-View-Controller program that creates these puzzles and allows its user to solve them, created a board in the Model class with all variables required to capture its state at any given time, used the Place class to access and modify the position of players, wrote methods to randomly generate new games in the Puzzle Generator class, and modified the Board Widget class to display the puzzle board.


## 12) CompSci 61A (Structure and Interpretation of Computer Programs) Projects

**(a) Scheme Interpreter:** developed a Python interpreter for a subset of the Scheme language. After examining the design of our target language, built an interpreter with functions that read Scheme expressions, evaluate them, and display the results. Used all knowledge learned from CompSci 61a to apply to this final project. 

**(b) Ants:** Replicated tower defense game Ants vs. Bees both GUI and command-line version. Used object-oriented programming to create, update gamestate, and move different ants and bee objects for ants to win the game with different classes, methods, attributes, objects, list comprehensions. Applied  abstraction, polymorphism, inheritance among other OOP concepts. 

**(c) CATS (CS61A AutoCorrected Typing Software):** Wrote program measuring typing speed. Additionally, implemented a feature to correct spelling of a word after a user types it. Some helper functions extracted relevant text from selected paragraphs, and were swap functions to add distances of non-matching elements, and compute (minimum) edit distance. Used concepts of recursion, higher-order functions, self-reference, abstraction, and structures including nested loops, dictionaries and list-comprehensions. 

**(d) Hog:** Developed a simulator and multiple strategies for the two-player dice game Hog. Helper functions implemented included applications of special rules like pig out, free bacon, feral hog, swine swap, and functions to simulate taking turns till the fastest player gets to the max score of 100 based on helper functions for score-maximizing strategies (highest average turn score). Commentary functions implemented also announced the players’ score after each turn, the lead score changes when applicable, and when a certain player’s score increases the highest during a specific game. Used higher-order functions and control statements, along with various applications of print, lambda, function calls and casting. 


# (B) Economics Projects' Description

This Github (SakshiSatpathy) has private code from homework, and projects from the following Economics classes taken at UC Berkeley: 
1) Econ 140 (Economic Statistics and Econometrics) 
2) Statistics 20 (Introduction to Probability and Statistics)

Selected Projects:

## 1) ECON 140 (Economic Statistics and Econometrics)
**Impact of Internet Access, Restaurant Workers, and Senior Citizens on Early COVID-19 Fatality rates [Language: Python (Pandas) and R]:** Evaluated the impact of internet access, magnitude of the service industry, and old age on early COVID-19 fatality rates in the US. Decided on a Multiple Linear Regression (MLR) model to jointly capture the impact of the natural log of all three variables on the log of proportion of COVID-19 cases in April 2020. The model was correctly specified after testing various relationships between the variables, ensuring roughly homoskedastic data, the expected error of our model being 0, with no perfect collinearity, and with attempt to introduce effective instrumental variables to reduce correlation between the variables and error term. 

## 2) Statistics 20 (Introduction to Probability and Statistics)
**(a) COVID-19 Pandemic Effects on Economy, Society, & Environment [Language: R]:** In regards to the economy, examined the housing market and change in rent prices. To analyze societal impacts, looked at many different characteristics such as the effectiveness of health insurances when many people require it at once, the effectiveness of lock-downs on controlling the spread of the virus, the negative side-effects of lockdowns such as depression, and how the lack of travel and mobility from the pandemic
has affected our environment. Used various statiscal inferences to detemine our conclusions on the pandemic, such as the hypothesis testing and regression.

**(b) Maternal Health and Smoking Relationship [Language: R]:** Visualized relationship between maternal health and smoking after accounting for confounding variables and limitations of observational studies conducted. Used ggplot functions to clean and subset the data, examine and analyze numerical summaries, and created various visualizations using geomplot.

For viewing code from these above projects, labs and homework, please send me requests at sakshi.satpathy@berkeley.edu so that I know you are not a student in the class. 
