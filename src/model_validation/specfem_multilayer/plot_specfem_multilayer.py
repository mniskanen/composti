# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

sem1 = np.loadtxt('MY.X1.FXZ.semv')
sem2 = np.loadtxt('MY.X2.FXZ.semv')
sem3 = np.loadtxt('MY.X3.FXZ.semv')
sem4 = np.loadtxt('MY.X4.FXZ.semv')
sem5 = np.loadtxt('MY.X5.FXZ.semv')
sem6 = np.loadtxt('MY.X6.FXZ.semv')
sem7 = np.loadtxt('MY.X7.FXZ.semv')
sem8 = np.loadtxt('MY.X8.FXZ.semv')
sem9 = np.loadtxt('MY.X9.FXZ.semv')
sem10 = np.loadtxt('MY.X10.FXZ.semv')


tshift = sem1[0, 0]
sem_time = sem1[:, 0] - tshift
sem_data = np.zeros((10, len(sem_time)))
sem_data[0] = sem1[:, 1]
sem_data[1] = sem2[:, 1]
sem_data[2] = sem3[:, 1]
sem_data[3] = sem4[:, 1]
sem_data[4] = sem5[:, 1]
sem_data[5] = sem6[:, 1]
sem_data[6] = sem7[:, 1]
sem_data[7] = sem8[:, 1]
sem_data[8] = sem9[:, 1]
sem_data[9] = sem10[:, 1]

# Change source direction 180 degrees...
sem_data = -sem_data

receivers = np.arange(3, 30 + 3, 3).astype('float64')
n_rec = len(receivers)

min_rec_distance = np.min(np.diff(receivers))
norm_coeff = 1 / np.max(np.abs(sem_data))



plt.figure()
for i in range(n_rec):
    plt.plot(receivers[i] + sem_data[i] * norm_coeff * min_rec_distance, sem_time, 'k-')

# plt.axis('tight')
plt.ylim([sem_time[-1], sem_time[0]])
# plt.gca().invert_yaxis()
plt.title('SPECFEM simulation')
plt.ylabel('Time (s)')
plt.xlabel('Receiver location and measurement (m)')
