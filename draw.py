import matplotlib.pyplot as plt
import numpy as np
three_epoch = np.linspace(1, 30, 30)
eight_epoch = np.linspace(1, 50, 50)
with open(r"C:\Users\admin\Desktop\eight_simple_train.txt", 'r') as file:
    data = file.read().splitlines()
eight_simple_train = [eval(i) for i in data]
eight_simple_train = np.asarray(eight_simple_train)

with open(r"C:\Users\admin\Desktop\eight_simple_valid.txt", 'r') as file:
    data = file.read().splitlines()
eight_simple_valid = [eval(i) for i in data]
eight_simple_valid = np.asarray(eight_simple_valid)

with open(r"C:\Users\admin\Desktop\three_simple_train.txt", 'r') as file:
    data = file.read().splitlines()
three_simple_train = [eval(i) for i in data]
three_simple_train = np.asarray(three_simple_train)

with open(r"C:\Users\admin\Desktop\three_simple_valid.txt", 'r') as file:
    data = file.read().splitlines()
three_simple_valid = [eval(i) for i in data]
three_simple_valid = np.asarray(three_simple_valid)

plt.figure(figsize = (12, 3), dpi = 72)
plt.subplots_adjust( bottom = 0.2, top = 0.9)

ax = plt.subplot(121)
line1, = plt.plot(three_epoch, three_simple_train, color = 'blue', linestyle = '-', linewidth = 3)
line2, = plt.plot(three_epoch, three_simple_valid, color = 'red', linestyle = '-', linewidth = 3)
vals = ax.get_yticks()
ax.set_yticklabels(['{:.0%}'.format(x) for x in vals])
plt.legend(handles = [line1,line2],labels = ['Training','Validation'],loc = 'best')
plt.xlabel('Epoch', fontsize = 14)
plt.ylabel('Accuracy', fontsize = 14)
# plt.xlim(xmin,xmax)
plt.tick_params(axis = 'both', which = 'major', labelsize = 10)
plt.title('Three type', fontsize = 14)
plt.grid('on')

ax2 = plt.subplot(122)
line3, = plt.plot(eight_epoch, eight_simple_train, color = 'blue', linestyle = '-', linewidth = 3)
line4, = plt.plot(eight_epoch, eight_simple_valid, color = 'red', linestyle = '-', linewidth = 3)
vals = ax2.get_yticks()
ax2.set_yticklabels(['{:.0%}'.format(x) for x in vals])
plt.legend(handles = [line3,line4],labels = ['Training','Validation'],loc = 'lower right')
plt.xlabel('Epoch', fontsize = 14)
plt.ylabel('Accuracy', fontsize = 14)
# plt.xlim(xmin,xmax)
plt.tick_params(axis = 'both', which = 'major', labelsize = 10)
plt.title('Eight type', fontsize = 14)
plt.grid('on')
plt.show()