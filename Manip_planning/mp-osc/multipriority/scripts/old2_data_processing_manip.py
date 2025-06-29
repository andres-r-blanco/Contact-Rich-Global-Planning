import matplotlib.pyplot as plt
import numpy as np
import os
import csv
TRIAL_NUM = 50 
SHORTEST_DIST_2D = 2.101
SHORTEST_DIST_3D = 4.0806000000000004
SHORTEST_DIST_3D_w_OBSTACLES = 4.0918

save_dir = r"C:\Users\arbxe\OneDrive\Desktop\code stuff\EmPRISE repos\Manip planning\mp-osc\multipriority\data\manip_data"

def main():
    file_path1 = os.path.join(save_dir, "reducing_objects_vs_contact_sample", 
        "rovcs_weight1_contactsamplechance0.0_objreduction0.0_Min Iterations1500")
    file_path2 = os.path.join(save_dir, "reducing_objects_vs_contact_sample", 
        "rovcs_weight1_contactsamplechance0.0_objreduction0.02_Min Iterations1500")
    file_path3 = os.path.join(save_dir, "reducing_objects_vs_contact_sample", 
        "rovcs_weight1_contactsamplechance0.2_objreduction0.0_Min Iterations1500")

    file_path_list = [file_path1,file_path2,file_path3]
    labels = ["Vanilla (no manip cost)","Object Reduction (no manip cost)","Contact Sampling (no manip cost)"]
    # # title = "Integral of Manipulability / Total Distance vs. Percent Total Distance (2D)"
    # # plot_manips_integral_and_total_dist(file_path_list,labels,title)
    # manip_threshold = 0.2
    # divisions = 4
    # title = f"Number of nodes with under {manip_threshold} manipulability vs distance divisions (3D with obstacles)"
    # plot_divided_manip_and_total_dist(manip_threshold, divisions, file_path_list,labels,title)

    plot_manip_over_dist(file_path1,labels[0])
    plot_manip_over_dist(file_path2,labels[1])
    plot_manip_over_dist(file_path3,labels[2])
    # mean_manip_over_weights(file_path1,file_path2,"Lazy comparison","Old Cost Avg Node Manipulability", "Old Cost Total Distance", "Not Lazy Avg Node Manipulability", "Not Lazy  Total Distance")
    

    # list_of_trials_init = [0]
    # add = TRIAL_NUM
    # mult = 1
    # list_of_trials = [x + add*mult for x in list_of_trials_init]

    # #old manip
    # file_path1 = r"C:\Users\arbxe\OneDrive\Desktop\code stuff\EmPRISE repos\Manip planning\mp-osc\multipriority\data\manip_data\2D\LazyFalse_Node cost functionTrue_Percent Manip Threshold0_Update PathFalse_Increasing weightFalse_Min Iterations1200"
    # #new manip
    # file_path2 = r"C:\Users\arbxe\OneDrive\Desktop\code stuff\EmPRISE repos\Manip planning\mp-osc\multipriority\data\manip_data\2D\LazyFalse_Node cost functionFalse_Percent Manip Threshold0_Update PathFalse_Increasing weightFalse_Min Iterations1200"
    # for trial_row in list_of_trials:
    #     [dist_data1,manip_data1] = get_distance_manip_pair_from_trial_row(file_path1, trial_row)
    #     [dist_data2,manip_data2] = get_distance_manip_pair_from_trial_row(file_path2, trial_row)

    
    
    # plt.errorbar(dist_data1, manip_data1, yerr=0, fmt='-o', color='blue', label="Old Cost",capsize=5)
    # plt.errorbar(dist_data2, manip_data2, yerr=0, fmt='-o', color='red', label="New Cost", capsize=5)
    # plt.xlabel("Distances")
    # plt.ylabel("Manipulability")
    # plt.title("Old Cost Vs New Cost Not Lazy 0.85")
    # plt.legend()
    # plt.show()

def mean_manip_per_distance_over_weights(divisions,file_path1, file_path2,test_type, label1,label2,label3,label4):
    list_of_trials_init = [0,10,20,30,40]
    add = TRIAL_NUM
    mult = 0
    [mean_manip_09,std_dev_manip_09,mean_dist_09,std_dev_dist_09] = get_mean_and_std_manip_and_dist([x + add*mult for x in list_of_trials_init],file_path1)
    mult = 1
    [mean_manip_085,std_dev_manip_085,mean_dist_085,std_dev_dist_085] = get_mean_and_std_manip_and_dist([x + add*mult for x in list_of_trials_init],file_path1)
    mult = 2
    [mean_manip_08,std_dev_manip_08,mean_dist_08,std_dev_dist_08] = get_mean_and_std_manip_and_dist([x + add*mult for x in list_of_trials_init],file_path1)
    mult = 3
    [mean_manip_07,std_dev_manip_07,mean_dist_07,std_dev_dist_07] = get_mean_and_std_manip_and_dist([x + add*mult for x in list_of_trials_init],file_path1)
    
    weights = [0.9,0.85,0.8,0.7]
    manip_means1 = [mean_manip_09,mean_manip_085,mean_manip_08,mean_manip_07]
    manip_stds1 = [std_dev_manip_09,std_dev_manip_085,std_dev_manip_08,std_dev_manip_07]
    dist_means1 = [mean_dist_09,mean_dist_085,mean_dist_08,mean_dist_07]
    dist_stds1 = [std_dev_dist_09,std_dev_dist_085,std_dev_dist_08,std_dev_dist_07]

    add = TRIAL_NUM
    mult = 0
    [mean_manip_09,std_dev_manip_09,mean_dist_09,std_dev_dist_09] = get_mean_and_std_manip_and_dist([x + add*mult for x in list_of_trials_init],file_path2)
    mult = 1
    [mean_manip_085,std_dev_manip_085,mean_dist_085,std_dev_dist_085] = get_mean_and_std_manip_and_dist([x + add*mult for x in list_of_trials_init],file_path2)
    mult = 2
    [mean_manip_08,std_dev_manip_08,mean_dist_08,std_dev_dist_08] = get_mean_and_std_manip_and_dist([x + add*mult for x in list_of_trials_init],file_path2)
    mult = 3
    [mean_manip_07,std_dev_manip_07,mean_dist_07,std_dev_dist_07] = get_mean_and_std_manip_and_dist([x + add*mult for x in list_of_trials_init],file_path2)

    manip_means2 = [mean_manip_09,mean_manip_085,mean_manip_08,mean_manip_07]
    manip_stds2 = [std_dev_manip_09,std_dev_manip_085,std_dev_manip_08,std_dev_manip_07]
    dist_means2 = [mean_dist_09,mean_dist_085,mean_dist_08,mean_dist_07]
    dist_stds2 = [std_dev_dist_09,std_dev_dist_085,std_dev_dist_08,std_dev_dist_07]

    title = ('Avg Node Manipulability and Distance Cost vs. Weight (' + test_type + ')')
    # plt_w_err(weights,manip_means,manip_stds, 'Distance weights','Costs',title,x2 = weights, y2 = dist_means,yerr2 = dist_stds, label1 = "Avg Node Manipulability Cost", label2="Total Distance Cost")
    plt4_w_err('Distance weights','Costs',title, weights, manip_means1, manip_stds1, label1, dist_means1, dist_stds1, label2, manip_means2, manip_stds2, label3, dist_means2, dist_stds2, label4)
    # plt_w_err(weights,manip_means,manip_stds, 'Distance weights','Manip Cost',('Avg Node Manipulability vs. Weight (' + test_type + ')'))
    # plt_w_err(weights,dist_means,dist_stds, 'Distance weights','Distance Cost',('Distance Cost vs. Weight (' + test_type + ')'))

def mean_manip_over_weights(file_path1, file_path2,test_type, label1,label2,label3,label4):
    list_of_trials_init = [0,10,20,30,40]
    add = TRIAL_NUM
    mult = 0
    [mean_manip_09,std_dev_manip_09,mean_dist_09,std_dev_dist_09] = get_mean_and_std_manip_and_dist([x + add*mult for x in list_of_trials_init],file_path1)
    mult = 1
    [mean_manip_085,std_dev_manip_085,mean_dist_085,std_dev_dist_085] = get_mean_and_std_manip_and_dist([x + add*mult for x in list_of_trials_init],file_path1)
    mult = 2
    [mean_manip_08,std_dev_manip_08,mean_dist_08,std_dev_dist_08] = get_mean_and_std_manip_and_dist([x + add*mult for x in list_of_trials_init],file_path1)
    mult = 3
    [mean_manip_07,std_dev_manip_07,mean_dist_07,std_dev_dist_07] = get_mean_and_std_manip_and_dist([x + add*mult for x in list_of_trials_init],file_path1)
    
    weights = [0.9,0.85,0.8,0.7]
    manip_means1 = [mean_manip_09,mean_manip_085,mean_manip_08,mean_manip_07]
    manip_stds1 = [std_dev_manip_09,std_dev_manip_085,std_dev_manip_08,std_dev_manip_07]
    dist_means1 = [mean_dist_09,mean_dist_085,mean_dist_08,mean_dist_07]
    dist_stds1 = [std_dev_dist_09,std_dev_dist_085,std_dev_dist_08,std_dev_dist_07]

    add = TRIAL_NUM
    mult = 0
    [mean_manip_09,std_dev_manip_09,mean_dist_09,std_dev_dist_09] = get_mean_and_std_manip_and_dist([x + add*mult for x in list_of_trials_init],file_path2)
    mult = 1
    [mean_manip_085,std_dev_manip_085,mean_dist_085,std_dev_dist_085] = get_mean_and_std_manip_and_dist([x + add*mult for x in list_of_trials_init],file_path2)
    mult = 2
    [mean_manip_08,std_dev_manip_08,mean_dist_08,std_dev_dist_08] = get_mean_and_std_manip_and_dist([x + add*mult for x in list_of_trials_init],file_path2)
    mult = 3
    [mean_manip_07,std_dev_manip_07,mean_dist_07,std_dev_dist_07] = get_mean_and_std_manip_and_dist([x + add*mult for x in list_of_trials_init],file_path2)

    manip_means2 = [mean_manip_09,mean_manip_085,mean_manip_08,mean_manip_07]
    manip_stds2 = [std_dev_manip_09,std_dev_manip_085,std_dev_manip_08,std_dev_manip_07]
    dist_means2 = [mean_dist_09,mean_dist_085,mean_dist_08,mean_dist_07]
    dist_stds2 = [std_dev_dist_09,std_dev_dist_085,std_dev_dist_08,std_dev_dist_07]

    title = ('Avg Node Manipulability and Distance Cost vs. Weight (' + test_type + ')')
    # plt_w_err(weights,manip_means,manip_stds, 'Distance weights','Costs',title,x2 = weights, y2 = dist_means,yerr2 = dist_stds, label1 = "Avg Node Manipulability Cost", label2="Total Distance Cost")
    plt4_w_err('Distance weights','Costs',title, weights, manip_means1, manip_stds1, label1, dist_means1, dist_stds1, label2, manip_means2, manip_stds2, label3, dist_means2, dist_stds2, label4)
    # plt_w_err(weights,manip_means,manip_stds, 'Distance weights','Manip Cost',('Avg Node Manipulability vs. Weight (' + test_type + ')'))
    # plt_w_err(weights,dist_means,dist_stds, 'Distance weights','Distance Cost',('Distance Cost vs. Weight (' + test_type + ')'))

def plt2_w_err(x,y,yerr, xlabel,ylabel,title, x2 = None, y2 = None,yerr2 = None, label1 = None, label2=None):
    if x2 is not None:
        plt.errorbar(x, y, yerr=yerr, fmt='-o', color='blue', label=label1,capsize=5)
        plt.errorbar(x2, y2, yerr=yerr2, fmt='-o', color='red', label=label2, capsize=5)
    else:
        plt.errorbar(x, y, yerr=yerr, fmt='-o', capsize=5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def plt4_w_err(xlabel,ylabel,title, x, y1, yerr1, label1, y2, yerr2, label2, y3, yerr3, label3, y4, yerr4, label4):
    plt.errorbar(x, y1, yerr=yerr1, fmt='-o', color='blue', label=label1, capsize=5)
    plt.errorbar(x, y2, yerr=yerr2, fmt='-o', color='cyan', label=label2, capsize=5)
    plt.errorbar(x, y3, yerr=yerr3, fmt='-o', color='red', label=label3, capsize=5)
    plt.errorbar(x, y4, yerr=yerr4, fmt='-o', color='orange', label=label4, capsize=5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

# def get_per_division_mean_and_std_manip_and_dist(division, list_of_trials,file_path):
#     manip_metrics = []
#     distance_metrics = []
#     for i in range(division):
#         manip_metrics.append([])
#     for trial_row in list_of_trials:
#         [dist_data,manip_data] = get_distance_manip_pair_from_trial_row(file_path, trial_row)
#         distance_metrics.append(sum(dist_data))
#         d = 0
#         for i in range(len(manip_data)):
#             if sum(dist_data[:i])>(sum(dist_data)*(d+1)/division):
#                 d += 1
            
#         manip_metric = 0
#         for m in manip_data:
#             manip_metric += 1/m
#         manip_metrics.append(manip_metric/len(manip_data))
#     mean_manip = np.mean(manip_metrics)
#     std_dev_manip = np.std(manip_metrics)
#     mean_dist = np.mean(distance_metrics)
#     std_dev_dist = np.std(distance_metrics)
#     return mean_manip, std_dev_manip, mean_dist, std_dev_dist


def plot_manip_over_dist(file_path, title):
    num_trials = 50
    colors = plt.cm.viridis(np.linspace(0, 1, num_trials))  # Generate distinct colors

    plt.figure(figsize=(10, 6))

    for trial_row in range(num_trials):
        try:
            dist_data, manip_data = get_distance_manip_pair_from_trial_row(file_path, trial_row)
            distances = cumulative_sum(dist_data)
            total_dist = distances[-1]
            percentage_distances = [x / total_dist for x in distances]
            plt.plot(percentage_distances, manip_data, color=colors[trial_row], alpha=0.6, label=f'Trial {trial_row}')
        except Exception as e:
            print(f"Skipping trial {trial_row}: {e}")
            continue

    plt.xlabel("Normalized Distance Along Path")
    plt.ylabel("Manipulability")
    plt.title(title)
    plt.tight_layout()
    plt.grid(True)
    plt.show()

# def plot_manips_over_dist(file_path,title):
#     list_of_trials_init = [0,10,20,30,40]
#     add = 50
#     colors = ['red', 'blue', 'green', 'purple', 'orange']
#     weights = [0.9,0.85,0.8,0.7]
    
#     fig, axs = plt.subplots(2, 2, figsize=(12, 10))
#     for mult in range(0, 4): 
#         list_of_trials = [x + add * (mult) for x in list_of_trials_init] 
#         ax = axs[(mult) // 2, (mult) % 2] # Determine subplot position 
#         for i, trial_row in enumerate(list_of_trials): 
#             [dist_data, manip_data] = get_distance_manip_pair_from_trial_row(file_path, trial_row) 
#             distances = cumulative_sum(dist_data) 
#             total_dist = distances[-1]
#             percentage_distances = [x / total_dist for x in distances]
#             ax.plot(percentage_distances, manip_data, color=colors[i], marker='o', label=f'Trial {i+1}')
#         ax.set_xlabel("distance") 
#         ax.set_ylabel("manipulability") 
#         ax.set_title(f"Dist weight = {weights[mult]}") 
#         ax.legend() 
#     plt.tight_layout() 
#     plt.suptitle(title)
#     plt.show()
#     return

def plot_manips_integral_and_total_dist(file_path_list,labels,title):
    list_of_trials_init = [0,10,20,30,40]
    add = TRIAL_NUM
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray', 'orange', 'purple']
    weights = [0.9,0.85,0.8,0.7]
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for mult in range(0, 4): 
        total_dist_list = []
        manip_trap_integral = []
        manip_trap_integral_over_dist = []
        list_of_trials = [x + add * mult for x in list_of_trials_init] 
        ax = axs[(mult) // 2, (mult) % 2] # Determine subplot position 
        for f, file_path in enumerate(file_path_list):
            for i, trial_row in enumerate(list_of_trials): 
                [dist_data, manip_data] = get_distance_manip_pair_from_trial_row(file_path, trial_row) 
                distances = cumulative_sum(dist_data) 
                total_dist = distances[-1]
                total_dist_list.append(total_dist/SHORTEST_DIST_2D)
                manip_trap_integral.append(trapezoidal_integral(distances,manip_data))
                manip_trap_integral_over_dist.append(trapezoidal_integral(distances,manip_data)/total_dist)
            # print(total_dist_list)
            mean_manip_trap_integral_over_dist = np.mean(manip_trap_integral_over_dist)
            std_dev_manip_trap_integral_over_dist = np.std(manip_trap_integral_over_dist)
            mean_total_dist = np.mean(total_dist_list)
            std_dev_total_dist = np.std(total_dist_list)
            ax.errorbar(mean_total_dist, mean_manip_trap_integral_over_dist, xerr=std_dev_total_dist, yerr=std_dev_manip_trap_integral_over_dist, fmt='-o', color=colors[f], label=labels[f],capsize=5)
        ax.set_xlabel("distance (%)") 
        ax.set_ylabel("manipulability") 
        ax.set_title(f"Dist weight = {weights[mult]}") 
        ax.legend() 
    plt.tight_layout() 
    plt.suptitle(title)
    plt.show()
    return

def plot_divided_manip_and_total_dist(manip_threshold, divisions, file_path_list,labels,title):
    list_of_trials_init = [0,10,20,30,40]
    add = TRIAL_NUM
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray', 'orange', 'purple']
    weights = [0.9,0.85,0.8,0.7]
    div_nums = list(range(1, divisions + 1))
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for mult in range(0, 4): 
        total_dist_list = []
        div_points_mean_list = []
        div_points_std_list = []
        list_of_trials = [x + add * mult for x in list_of_trials_init] 
        div_point_matrix = np.zeros((len(list_of_trials),divisions))
        ax = axs[(mult) // 2, (mult) % 2] # Determine subplot position 
        for f, file_path in enumerate(file_path_list):
            for i, trial_row in enumerate(list_of_trials): 
                [dist_data, manip_data] = get_distance_manip_pair_from_trial_row(file_path, trial_row) 
                distances = cumulative_sum(dist_data) 
                total_dist = distances[-1]
                total_dist_list.append(total_dist/SHORTEST_DIST_2D)
                dist_idx = 0
                for j,manip in enumerate(manip_data):
                    if distances[j] > ((dist_idx+1)*(total_dist/4)):
                        dist_idx += 1
                    if manip < manip_threshold:
                        div_point_matrix[i,dist_idx] += 1
            div_points_mean = np.zeros(divisions)
            div_points_std = np.zeros(divisions)
            for i in range(0,div_point_matrix.shape[1]):
                div_points_mean[i] = np.mean(div_point_matrix[:,i])
                div_points_std[i] = np.std(div_point_matrix[:,i])
            ax.errorbar(div_nums, div_points_mean, yerr=div_points_std, fmt='-o', color=colors[f], label=labels[f],capsize=4)
        # ax.set_xlabel("Distance division") 
        # ax.set_ylabel(f"Number of nodes below manip threshold of {manip_threshold}") 
        ax.set_title(f"Dist weight = {weights[mult]}") 
        if mult == 3:
            ax.legend(loc="upper center")
        # ax.legend() 
    # plt.tight_layout() 
    plt.suptitle(title)
    plt.show()
    return
    

def get_mean_and_std_manip_and_dist(list_of_trials,file_path):
    manip_metrics = []
    distance_metrics = []
    for trial_row in list_of_trials:
        [dist_data,manip_data] = get_distance_manip_pair_from_trial_row(file_path, trial_row)
        distance_metrics.append(sum(dist_data))
        manip_metric = 0
        for m in manip_data:
            manip_metric += 1/m
        manip_metrics.append(manip_metric/len(manip_data))
    mean_manip = np.mean(manip_metrics)
    std_dev_manip = np.std(manip_metrics)
    mean_dist = np.mean(distance_metrics)
    std_dev_dist = np.std(distance_metrics)
    return mean_manip, std_dev_manip, mean_dist, std_dev_dist
    
def get_mean_and_std_manip_and_dist(list_of_trials,file_path):
    manip_metrics = []
    distance_metrics = []
    for trial_row in list_of_trials:
        [dist_data,manip_data] = get_distance_manip_pair_from_trial_row(file_path, trial_row)
        distance_metrics.append(sum(dist_data))
        manip_metric = 0
        for m in manip_data:
            manip_metric += 1/m
        manip_metrics.append(manip_metric/len(manip_data))
    mean_manip = np.mean(manip_metrics)
    std_dev_manip = np.std(manip_metrics)
    mean_dist = np.mean(distance_metrics)
    std_dev_dist = np.std(distance_metrics)
    return mean_manip, std_dev_manip, mean_dist, std_dev_dist


def get_distance_manip_pair_from_trial_row(file_path, row_num):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        manip_row = rows[row_num+1]
        dist_row = rows[row_num+2]
        manip_list = list(map(float, manip_row[1:]))
        dist_row = list(map(float, dist_row[1:]))
        manip_list.pop()
        dist_row.pop()
        return [dist_row,manip_list]
    
def cumulative_sum(dist_data): 
    distances = [] 
    cumulative = 0 
    for value in dist_data: 
        cumulative += value 
        distances.append(cumulative) 
    return distances

def trapezoidal_integral(x, y): 
    integral = 0 
    for i in range(1, len(x)): 
        dx = x[i] - x[i-1] 
        integral += (y[i] + y[i-1]) * dx / 2 
    return integral

if __name__ == '__main__':
    main()