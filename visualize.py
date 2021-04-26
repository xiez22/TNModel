import matplotlib.pyplot as plt
import seaborn as sns
import os


log_dir = 'logs'

loss_dict = {}
acc_dict = {}


for root, dirs, files in os.walk(log_dir):
    for f in files:
        f_path = os.path.join(root, f)

        info = f_path.split('/')[1].split('.')[0].split('_')

        if info[-1] == 'acc':
            acc_dict[info[2]] = [], []
            fi = open(f_path, 'rt')

            for line in fi.readlines():
                a, b = line.split(',')
                acc_dict[info[2]][0].append(int(a))
                acc_dict[info[2]][1].append(float(b))

            fi.close()
        else:
            loss_dict[info[2]] = [], []
            fi = open(f_path, 'rt')

            for line in fi.readlines():
                a, b = line.split(',')
                loss_dict[info[2]][0].append(int(a))
                loss_dict[info[2]][1].append(float(b))


def smooth(a, weight=0.6):
    for i in range(1, len(a)):
        a[i] = weight * a[i-1] + (1.0 - weight) * a[i]


# Loss Figure
loss_list = sorted(loss_dict.items())

sns.set()
for k, v in loss_list:
    smooth(v[1])
    plt.plot(v[0], v[1], label=f'bond_dim={k}')

plt.legend()
plt.title('Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.show()


acc_list = sorted(acc_dict.items())

# Acc Figure
for k, v in acc_list:
    smooth(v[1])
    plt.plot(v[0], v[1], label=f'bond_dim={k}')

plt.legend()
plt.title('Accuracy')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.show()
