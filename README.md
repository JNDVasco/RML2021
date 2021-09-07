# RML2021


# Módulo 1 - Introduction and overview: robotics and machine learning 

# Módulo 2 - Fundamentals of mobile robotics 

# Módulo 3 - Mobile robotics hardware and systems

# Módulo 4 - Mobile robots perception 

# Módulo 5 - Robotics simulation and Robot Operating System

Documentação deste módulo [aqui](Módulo%205%20-%20ROS/ROS.md)

# Módulo 6 - Probability, Bayesian inference and Bayesian Networks

# Módulo 7 - Pattern recognition, principles and algorithms

## Vote

### Naive Bayes com seleção de atributos

```
Correctly Classified Instances         418               96.092  %
Incorrectly Classified Instances        17                3.908  %
Kappa statistic                          0.9177
Mean absolute error                      0.0575
Root mean squared error                  0.1768
Relative absolute error                 12.1285 %
Root relative squared error             36.3023 %
Total Number of Instances              435     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0,966    0,048    0,970      0,966    0,968      0,918    0,991     0,995     democrat
                 0,952    0,034    0,947      0,952    0,950      0,918    0,991     0,984     republican
Weighted Avg.    0,961    0,042    0,961      0,961    0,961      0,918    0,991     0,991     

=== Confusion Matrix ===

   a   b   <-- classified as
 258   9 |   a = democrat
   8 160 |   b = republican
```
### Random Forest

```
Correctly Classified Instances         418               96.092  %
Incorrectly Classified Instances        17                3.908  %
Kappa statistic                          0.9175
Mean absolute error                      0.0714
Root mean squared error                  0.1742
Relative absolute error                 15.0587 %
Root relative squared error             35.7776 %
Total Number of Instances              435     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0,970    0,054    0,966      0,970    0,968      0,917    0,993     0,996     democrat
                 0,946    0,030    0,952      0,946    0,949      0,917    0,993     0,988     republican
Weighted Avg.    0,961    0,044    0,961      0,961    0,961      0,917    0,993     0,993     

=== Confusion Matrix ===

   a   b   <-- classified as
 259   8 |   a = democrat
   9 159 |   b = republican
```
### Random Forest com seleção de atributos

```
Correctly Classified Instances         418               96.092  %
Incorrectly Classified Instances        17                3.908  %
Kappa statistic                          0.9177
Mean absolute error                      0.0594
Root mean squared error                  0.1714
Relative absolute error                 12.524  %
Root relative squared error             35.1995 %
Total Number of Instances              435     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0,966    0,048    0,970      0,966    0,968      0,918    0,989     0,993     democrat
                 0,952    0,034    0,947      0,952    0,950      0,918    0,989     0,975     republican
Weighted Avg.    0,961    0,042    0,961      0,961    0,961      0,918    0,989     0,986     

=== Confusion Matrix ===

   a   b   <-- classified as
 258   9 |   a = democrat
   8 160 |   b = republican
```

## Sonar

### Random Forest

```
Correctly Classified Instances         169               81.25   %
Incorrectly Classified Instances        39               18.75   %
Kappa statistic                          0.6191
Mean absolute error                      0.3079
Root mean squared error                  0.3547
Relative absolute error                 61.8568 %
Root relative squared error             71.1001 %
Total Number of Instances              208     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0,711    0,099    0,863      0,711    0,780      0,628    0,931     0,923     Rock
                 0,901    0,289    0,781      0,901    0,837      0,628    0,931     0,943     Mine
Weighted Avg.    0,813    0,200    0,819      0,813    0,810      0,628    0,931     0,933     

=== Confusion Matrix ===

   a   b   <-- classified as
  69  28 |   a = Rock
  11 100 |   b = Mine
```

## Letters

### LibSVM

```
Correctly Classified Instances       19515               97.575  %
Incorrectly Classified Instances       485                2.425  %
Kappa statistic                          0.9748
Mean absolute error                      0.0019
Root mean squared error                  0.0432
Relative absolute error                  2.5221 %
Root relative squared error             22.4592 %
Total Number of Instances            20000     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0,999    0,000    0,995      0,999    0,997      0,997    0,999     0,994     A
                 0,974    0,003    0,937      0,974    0,955      0,954    0,986     0,914     B
                 0,973    0,000    0,990      0,973    0,981      0,981    0,986     0,964     C
                 0,976    0,003    0,939      0,976    0,957      0,956    0,987     0,918     D
                 0,977    0,001    0,970      0,977    0,973      0,972    0,988     0,948     E
                 0,969    0,001    0,972      0,969    0,970      0,969    0,984     0,943     F
                 0,969    0,001    0,971      0,969    0,970      0,969    0,984     0,943     G
                 0,926    0,002    0,944      0,926    0,935      0,933    0,962     0,878     H
                 0,956    0,001    0,974      0,956    0,965      0,964    0,978     0,933     I
                 0,961    0,002    0,957      0,961    0,959      0,958    0,980     0,922     J
                 0,955    0,001    0,964      0,955    0,960      0,958    0,977     0,923     K
                 0,979    0,000    0,992      0,979    0,985      0,985    0,989     0,972     L
                 0,989    0,001    0,985      0,989    0,987      0,986    0,994     0,974     M
                 0,973    0,001    0,983      0,973    0,978      0,977    0,986     0,958     N
                 0,976    0,001    0,970      0,976    0,973      0,972    0,987     0,947     O
                 0,961    0,001    0,985      0,961    0,973      0,972    0,980     0,948     P
                 0,992    0,001    0,981      0,992    0,987      0,986    0,996     0,974     Q
                 0,959    0,002    0,945      0,959    0,952      0,950    0,978     0,908     R
                 0,993    0,000    0,995      0,993    0,994      0,994    0,997     0,988     S
                 0,985    0,000    0,994      0,985    0,989      0,989    0,992     0,979     T
                 0,993    0,000    0,990      0,993    0,991      0,991    0,996     0,983     U
                 0,965    0,001    0,985      0,965    0,975      0,974    0,982     0,952     V
                 0,989    0,000    0,988      0,989    0,989      0,988    0,994     0,978     W
                 0,989    0,001    0,980      0,989    0,984      0,984    0,994     0,969     X
                 0,992    0,000    0,992      0,992    0,992      0,992    0,996     0,985     Y
                 0,993    0,000    0,990      0,993    0,992      0,992    0,996     0,984     Z
Weighted Avg.    0,976    0,001    0,976      0,976    0,976      0,975    0,987     0,953     

=== Confusion Matrix ===

   a   b   c   d   e   f   g   h   i   j   k   l   m   n   o   p   q   r   s   t   u   v   w   x   y   z   <-- classified as
 788   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0 |   a = A
   0 746   0   2   1   0   0   3   0   0   1   0   0   0   0   0   0   4   1   0   1   5   0   2   0   0 |   b = B
   0   0 716   0   6   0   5   1   0   0   0   1   0   0   5   0   0   0   0   0   0   0   2   0   0   0 |   c = C
   0   2   0 786   0   0   0   9   0   0   0   0   0   4   3   0   0   1   0   0   0   0   0   0   0   0 |   d = D
   0   2   0   0 750   1   6   0   0   0   0   3   0   0   0   0   0   0   0   0   0   0   0   0   0   6 |   e = E
   0   1   0   2   2 751   0   1   0   1   0   0   0   0   0   9   0   0   1   3   0   2   0   1   0   1 |   f = F
   0   2   1   9   4   0 749   0   0   0   0   0   1   0   3   0   0   0   1   0   0   1   2   0   0   0 |   g = G
   0   2   0  15   1   0   5 680   0   1  13   0   1   0   1   0   1  12   0   0   2   0   0   0   0   0 |   h = H
   0   0   1   1   0   0   0   0 722  29   0   0   0   1   0   0   0   0   0   0   0   0   0   1   0   0 |   i = I
   1   0   0   2   0   0   0   1  19 718   0   0   0   2   3   0   0   0   0   0   1   0   0   0   0   0 |   j = J
   0   0   0   0   1   0   0   7   0   0 706   0   0   0   0   0   0  15   0   0   1   0   0   9   0   0 |   k = K
   0   1   0   0   4   0   2   3   0   1   2 745   0   0   0   0   0   1   1   0   0   0   0   1   0   0 |   l = L
   0   5   0   0   0   0   2   0   0   0   0   0 783   1   0   0   0   0   0   0   0   0   1   0   0   0 |   m = M
   0   0   0   2   0   0   0   5   0   0   0   0   3 762   4   0   0   6   0   0   0   1   0   0   0   0 |   n = N
   0   0   4   9   0   0   0   0   0   0   0   0   1   0 735   0   2   0   0   0   1   0   1   0   0   0 |   o = O
   0   2   0   0   1  18   0   3   0   0   0   1   0   0   0 772   3   0   0   0   0   0   1   0   2   0 |   p = P
   1   0   0   1   0   0   0   0   0   0   0   0   0   0   3   0 777   1   0   0   0   0   0   0   0   0 |   q = Q
   0  12   0   3   0   0   0   3   0   0   6   0   0   5   0   0   2 727   0   0   0   0   0   0   0   0 |   r = R
   0   1   0   0   2   0   0   1   0   0   0   0   0   0   0   0   0   1 743   0   0   0   0   0   0   0 |   s = S
   0   1   1   3   0   1   0   1   0   0   0   1   0   0   0   1   0   0   0 784   0   0   0   1   2   0 |   t = T
   1   0   0   0   0   0   0   1   0   0   2   0   1   0   0   0   0   0   0   0 807   1   0   0   0   0 |   u = U
   0  18   0   0   0   2   1   0   0   0   0   0   1   0   0   2   0   0   0   0   0 737   2   0   1   0 |   v = V
   0   0   0   0   0   0   1   1   0   0   0   0   4   0   1   0   0   0   0   0   0   1 744   0   0   0 |   w = W
   0   0   0   2   1   0   0   0   0   0   2   0   0   0   0   0   2   1   0   1   0   0   0 778   0   0 |   x = X
   1   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   2   0   0   1 780   0 |   y = Y
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   5   0   0   0   0   0   0   0   0 729 |   z = Z
```

# Módulo 8 - Linear and non-linear regression

# Módulo 9 - Introduction to Deep-learning and machine vision

# Módulo 10 - Kalman filtering, Intro to Reinforcement Learning, Fuzzy-systems 
