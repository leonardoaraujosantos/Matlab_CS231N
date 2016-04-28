# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:02:17 2016

@author: leo
"""

import numpy as np
import cs231n.layers


scores = np.array([ [-2.85, 0.86, 0.28], [0, 0.3, 0.2], [0.3, -0.3, 0] ]);
# Don't forget the arrays on python start at zero...
correct = np.array([2, 1, 0]);

print('\nInput scores');
print(scores)
print('\nCorrect indexes');
print (correct)

# Call softmax_loss reference
result = cs231n.layers.softmax_loss(scores, correct)

print('\nLoss:')
print(result[0])

print('\nDw:')
print(result[1])
