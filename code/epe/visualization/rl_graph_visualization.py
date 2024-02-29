import argparse
import os
import json
from matplotlib import pyplot as plt
plt.style.use('bmh')

parser = argparse.ArgumentParser(description='This script can be used to visualize the reinforcement learning json final results of one or more runs (for comparison).')
parser.add_argument('--json_path', action='store', help='The path where the json files are stored.')

args = parser.parse_args()

if (args.json_path is None) or not os.path.isdir(args.json_path):
    print('--json_path argument is not set. Please provide a valid path in the disk where the json files are stored.')
    exit(1)

def plot_result_single(x,title,y_label,x_label):
    plt.plot(x)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()
def plot_result(x,type,title,y_label,x_label,legends,already_averaged):
    for i in range(len(x)):
        if already_averaged == False:
            if type == "actor_losses" or type == "critic_losses" or type == "epsilons":
                plt.plot([sum(sublist) / len(sublist) for sublist in data[i][type]])
            elif type == "distances":
                plt.plot([max(sublist) for sublist in data[i][type]])
            else:
                plt.plot(x[i][type])
        else:
            plt.plot(x[i][type])
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(legends, loc='upper left')
    plt.show()

data = []
names = []
for root, dirs, files in os.walk(os.path.abspath(args.json_path)):
    for file in files:
        data.append(json.load(open(os.path.join(root, file))))
        names.append(file.split(".")[0])

load_singe = False
already_averaged = False
if load_singe:
    for i in range(len(data)):
        plot_result_single(data[i]['steps'],"Ticks Graph","Ticks Count","Episode")
        plot_result_single(data[i]['rewards'],"Rewards Graph","Total Reward","Episode")
        plot_result_single([sum(sublist) / len(sublist) for sublist in data[i]['actor_losses']],"Actor Loss Graph","Average Loss","Episode")
        plot_result_single([sum(sublist) / len(sublist) for sublist in data[i]['critic_losses']],"Critic Loss Graph","Average Loss","Episode")
        plot_result_single([max(sublist) for sublist in data[i]['distances']], "Distances Graph", "Max Distance", "Episode")
        plot_result_single([sum(sublist) / len(sublist) for sublist in data[i]['epsilons']], "Epsilon Graph","Average Epsilon", "Episode")

else:
    plot_result(data,'steps',"Ticks Graph","Ticks Count","Episode",names,already_averaged)
    plot_result(data,'rewards',"Rewards Graph","Total Reward","Episode", names,already_averaged)
    plot_result(data,"actor_losses","Loss Graph","Average Loss","Episode", names,already_averaged)
    plot_result(data,"critic_losses","Critic Loss Graph","Average Loss","Episode", names,already_averaged)
    plot_result(data,"distances","Distances Graph","Max Distance","Episode", names,already_averaged)
    plot_result(data, "epsilons", "Epsilon Graph", "Average Epsilon", "Episode", names, already_averaged)
# Closing file
f.close()
