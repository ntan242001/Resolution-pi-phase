import numpy as np
import matplotlib.pyplot as plt
import csv
xmax = 1.5
xticks = np.linspace(-xmax,xmax,7)
xticks = np.rint(xticks)
xticks = list(xticks)

E = 15010  # eV  Nominal Energy After Acceleration
E_0 = 11.5 # eV  Energy at the sample ##########CUSTOMIZABLE INPUT##########
kappa = np.sqrt(E/E_0)

C_3 = 0.0297 * kappa**(1/2) + 0.1626  # m  Third Order Spherical Aberration Coefficient
C_5 = 0.5624 * kappa**(3/2) - 16.541  # m  Fifth Order Spherical Aberration Coefficient for ac LEEM

lamda = 6.6261e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * E) # in metre

delta_z_series1 = []

resolution_list1 = []

with open('R(dz) Gaussian spread Step phase object nac_LEEM_E0=11.5.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                delta_z_series1.append(float(row[0])*1e-6)
                resolution_list1.append(float(row[1]))    
    csvfile.close()
    
delta_z_series2 = []

resolution_list2 = []

with open('R(dz) FN spread Step phase object nac_LEEM_E0=11.5.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                delta_z_series2.append(float(row[0])*1e-6)
                resolution_list2.append(float(row[1]))    
    csvfile.close()
    
delta_z_series3 = []
resolution_list3 = []

with open('R(dz) Gauss G1 spread Step phase object nac_LEEM_E0=11.5.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                delta_z_series3.append(float(row[0])*1e-6)
                resolution_list3.append(float(row[1]))    
    csvfile.close()
  
delta_z_series4 = []
resolution_list4 = []

with open('R(dz) Gaussian spread Step phase object ac_LEEM_E0=11.5.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                delta_z_series4.append(float(row[0])*1e-6)
                resolution_list4.append(float(row[1]))    
    csvfile.close()
    
    
delta_z_series5 = []
resolution_list5 = []

with open('R(dz) FN spread Step phase object ac_LEEM_E0=11.5.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                delta_z_series5.append(float(row[0])*1e-6)
                resolution_list5.append(float(row[1]))    
    csvfile.close()
    
    
delta_z_series6 = []
resolution_list6 = []

with open('R(dz) Gauss G1 spread Step phase object ac_LEEM_E0=11.5.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                delta_z_series6.append(float(row[0])*1e-6)
                resolution_list6.append(float(row[1]))    
    csvfile.close()
    
size=5

plt.scatter(delta_z_series1/(C_3*lamda)**(1/2), resolution_list1, label = 'Gaussian', s=size, c='b', marker='o')
# plt.scatter(delta_z_series2/(C_3*lamda)**(1/2), resolution_list2, label = 'FN', s=20, c='b', marker='o')
# plt.scatter(delta_z_series3/(C_3*lamda)**(1/2), resolution_list3, label = 'G1', s=20, c='b', marker='o')

plt.scatter(delta_z_series4/(C_5*lamda**2)**(1/3), resolution_list4, label = 'acGaussian', s=size, c='r', marker='o')
# plt.scatter(delta_z_series5/(C_5*lamda**2)**(1/3), resolution_list5, label = 'acFN', s=20, c='b', marker='o')
# plt.scatter(delta_z_series6/(C_5*lamda**2)**(1/3), resolution_list6, label = 'acG1', s=20, c='b', marker='o')


plt.ylim(0, 6)
plt.xlim(-2, 2)

# plt.xticks(list(xticks))

# plt.vlines(x=1, ymin=0, ymax=6)

plt.title('$\pi$ phase')

# naming the x axis
plt.text(-0.8,-1,'$\Delta z ( $', color='k', fontsize=15)
plt.text(-0.5,-1,'$(C_3 \lambda)^{1/2} $', color='b', fontsize=13)
plt.text(0.05,-1,', ', color='k', fontsize=15)
plt.text(0.12,-1,'$(C_5 \lambda^2)^{1/3} $', color='r', fontsize=13)
plt.text(0.8,-1,' )', color='k', fontsize=15)

# naming the y axis
plt.ylabel('Resolution (nm)', fontsize=12)
plt.legend(bbox_to_anchor=(1, 1))

plt.show()
