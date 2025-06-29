import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def parse_file(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        data_rows = list(reader)

    data = [item for row in data_rows for item in row]  # Flatten list of rows into one list

    trials = {}
    i = 0
    while i < len(data):
        if data[i] == 'Trial':
            trial_num = int(data[i+1])
            trials[trial_num] = {}
            i += 2
            while i < len(data) and data[i] != 'Trial':
                key = data[i]
                values = []
                i += 1
                while i < len(data) and data[i] not in ['Trial', 'Time (s)', 'Joint Norm Distance', 'Manipulability', 'Distance to Taxel', 'Closest Taxel ID', 'Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Joint 7']:
                    try:
                        values.append(float(data[i]))
                        i += 1
                    except:
                        break
                trials[trial_num][key] = np.array(values)
        else:
            i += 1
    return trials

def ensure_plot_dir(base_path):
    plot_dir = os.path.join(base_path, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

def plot_manip_vs_joint_norm(trials, trial_num, base_path, show_plot=False):
    manip = trials[trial_num]['Manipulability']
    joint_norm = trials[trial_num]['Joint Norm Distance']

    plt.figure()
    plt.plot(joint_norm, manip, marker='o')
    plt.title(f'Trial {trial_num}: Manipulability vs Joint Norm Distance')
    plt.xlabel('Joint Norm Distance')
    plt.ylabel('Manipulability')
    plt.grid()

    plot_dir = ensure_plot_dir(base_path)
    filename = os.path.join(plot_dir, f'Trial_{trial_num}_Manip_vs_JointNorm.png')
    plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.close()

def plot_manip_vs_normalized_joint_norm_all(trials, base_path, show_plot=False):
    plt.figure()
    for trial_num, data_dict in trials.items():
        joint_norm = data_dict['Joint Norm Distance']
        manip = data_dict['Manipulability']
        cumulative_distance = np.cumsum(joint_norm)
        normalized_distance = cumulative_distance / cumulative_distance[-1]
        plt.plot(normalized_distance, manip, label=f'Trial {trial_num}')
    plt.title('Manipulability vs Normalized Joint Norm Distance (All Trials)')
    plt.xlabel('Normalized Cumulative Joint Norm Distance')
    plt.ylabel('Manipulability')
    plt.legend()
    plt.grid()

    plot_dir = ensure_plot_dir(base_path)
    filename = os.path.join(plot_dir, 'AllTrials_Manip_vs_NormalizedJointNorm.png')
    plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.close()

def plot_joint_norm_over_time(trials, trial_num, base_path, show_plot=False):
    time = trials[trial_num]['Time (s)']
    joint_norm = trials[trial_num]['Joint Norm Distance']

    plt.figure()
    plt.plot(time, joint_norm, marker='o')
    plt.title(f'Trial {trial_num}: Joint Norm Distance over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Norm Distance')
    plt.grid()

    plot_dir = ensure_plot_dir(base_path)
    filename = os.path.join(plot_dir, f'Trial_{trial_num}_JointNorm_vs_Time.png')
    plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.close()

def plot_total_joint_norm_per_trial(trials, base_path, show_plot=False):
    trial_ids = []
    total_distances = []
    for trial_num, data_dict in trials.items():
        total_distance = np.sum(data_dict['Joint Norm Distance'])
        trial_ids.append(trial_num)
        total_distances.append(total_distance)

    plt.figure()
    plt.bar(trial_ids, total_distances)
    plt.title('Total Joint Norm Distance per Trial')
    plt.xlabel('Trial')
    plt.ylabel('Total Joint Norm Distance')
    plt.grid(axis='y')

    plot_dir = ensure_plot_dir(base_path)
    filename = os.path.join(plot_dir, 'Total_JointNorm_per_Trial.png')
    plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.close()
    
def plot_manipulability_histogram(trials, base_path, show_plot=False):
    all_manip = np.concatenate([trials[trial]['Manipulability'] for trial in trials])
    plt.figure()
    plt.hist(all_manip, bins=50, edgecolor='black')
    plt.title('Histogram of Manipulability Across All Trials')
    plt.xlabel('Manipulability')
    plt.ylabel('Frequency')
    plt.grid()

    plot_dir = ensure_plot_dir(base_path)
    filename = os.path.join(plot_dir, 'Manipulability_Histogram.png')
    plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.close()

def plot_distance_vs_avg_manipulability(trials, base_path, show_plot=False):
    total_distances = []
    avg_manips = []
    for trial_num in trials:
        manip = trials[trial_num]['Manipulability']
        joint_norm = trials[trial_num]['Joint Norm Distance']
        total_distance = np.sum(joint_norm)
        avg_manip = np.mean(manip)
        total_distances.append(total_distance)
        avg_manips.append(avg_manip)

    plt.figure()
    plt.scatter(total_distances, avg_manips)
    plt.title('Total Distance vs Average Manipulability')
    plt.xlabel('Total Joint Norm Distance')
    plt.ylabel('Average Manipulability')
    plt.grid()

    plot_dir = ensure_plot_dir(base_path)
    filename = os.path.join(plot_dir, 'Distance_vs_AvgManipulability.png')
    plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.close()

def plot_manipulability_boxplot(trials, base_path, show_plot=False):
    data = [trials[trial]['Manipulability'] for trial in trials]
    labels = [f'Trial {trial}' for trial in trials]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title('Boxplot of Manipulability Across Trials')
    plt.xlabel('Trial')
    plt.ylabel('Manipulability')
    plt.xticks(rotation=45)
    plt.grid()

    plot_dir = ensure_plot_dir(base_path)
    filename = os.path.join(plot_dir, 'Manipulability_Boxplot.png')
    plt.savefig(filename, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()

def compute_trial_metrics(trials, trial_num):
    data = trials[trial_num]
    manip = data['Manipulability']
    joint_norm = data['Joint Norm Distance']
    time = data['Time (s)']

    avg_manip = np.mean(manip)
    num_low_manip = np.sum(manip < 0.01)
    total_distance = np.sum(joint_norm)
    total_time = time[-1] - time[0] if len(time) > 1 else 0.0

    return avg_manip, num_low_manip, total_distance, total_time

def compute_average_metrics_across_trials(trials, base_path, print_output=False):
    avg_manips = []
    num_low_manips = []
    total_distances = []
    total_times = []
    percentages_low_manip = []

    for trial_num in trials:
        avg_manip, num_low_manip, total_distance, total_time = compute_trial_metrics(trials, trial_num)
        avg_manips.append(avg_manip)
        num_low_manips.append(num_low_manip)
        total_distances.append(total_distance)
        total_times.append(total_time)

        # Calculate percentage of states with manipulability < 0.01
        total_states = len(trials[trial_num]['Manipulability'])
        percentage_low_manip = (num_low_manip / total_states) * 100 if total_states > 0 else 0
        percentages_low_manip.append(percentage_low_manip)

    mean_avg_manip = np.mean(avg_manips)
    mean_num_low_manip = np.mean(num_low_manips)
    mean_total_distance = np.mean(total_distances)
    mean_total_time = np.mean(total_times)
    mean_percentage_low_manip = np.mean(percentages_low_manip)

    std_avg_manip = np.std(avg_manips)
    std_num_low_manip = np.std(num_low_manips)
    std_total_distance = np.std(total_distances)
    std_total_time = np.std(total_times)
    std_percentage_low_manip = np.std(percentages_low_manip)

    # Save metrics to a CSV file
    metrics_file = os.path.join(base_path, "average_metrics.csv")
    os.makedirs(base_path, exist_ok=True)  # Ensure the directory exists
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Mean", "Standard Deviation"])
        writer.writerow(["Average Manipulability", mean_avg_manip, std_avg_manip])
        writer.writerow(["Number of states with manipulability < 0.01", mean_num_low_manip, std_num_low_manip])
        writer.writerow(["Percentage of states with manipulability < 0.01", mean_percentage_low_manip, std_percentage_low_manip])
        writer.writerow(["Total Joint Norm Distance", mean_total_distance, std_total_distance])
        writer.writerow(["Total Time", mean_total_time, std_total_time])

    if print_output:
        print("\nAverage metrics across all trials:")
        print(f"Average Manipulability: {mean_avg_manip:.4f} ± {std_avg_manip:.4f}")
        print(f"Number of states with manipulability < 0.01: {mean_num_low_manip:.2f} ± {std_num_low_manip:.2f}")
        print(f"Percentage of states with manipulability < 0.01: {mean_percentage_low_manip:.2f}% ± {std_percentage_low_manip:.2f}%")
        print(f"Total Joint Norm Distance: {mean_total_distance:.4f} ± {std_total_distance:.4f}")
        print(f"Total Time: {mean_total_time:.4f} ± {std_total_time:.4f}")

    return mean_avg_manip, std_avg_manip, mean_num_low_manip, std_num_low_manip, mean_percentage_low_manip, std_percentage_low_manip, mean_total_distance, std_total_distance, mean_total_time, std_total_time

def data_processing_pipeline(file_path, print_output=False,show_plot=False):
    trials = parse_file(file_path)
    base_path = file_path.rsplit('.', 1)[0]
    compute_average_metrics_across_trials(trials, base_path, print_output)

    plot_manip_vs_normalized_joint_norm_all(trials, base_path, show_plot=show_plot)
    plot_total_joint_norm_per_trial(trials, base_path, show_plot=show_plot)
    plot_manipulability_histogram(trials, base_path, show_plot=show_plot)
    plot_distance_vs_avg_manipulability(trials, base_path, show_plot=show_plot)
    plot_manipulability_boxplot(trials, base_path, show_plot=show_plot)

# Example usage:
if __name__ == "__main__":
    w_list = [1, 0.8]
    obj_reduction_list = [0.0, 0.02]
    DATA_PATH = "/home/rishabh/Andres/Manip_planning/mp-osc/multipriority/data/manip_data/mpc_new_reach_over_body"

    for w in w_list:
        for obj_reduction in obj_reduction_list:
            file_name = f"nrob_mpc_weight{w}_contactsamplechance0.0_objreduction{obj_reduction}_Min Iterations4000.csv"
            file_path = os.path.join(DATA_PATH, file_name)
            data_processing_pipeline(file_path, print_output=True,show_plot=False)
            
    file_name = f"nrob_mpc_no_planning_ee.csv"
    file_path = os.path.join(DATA_PATH, file_name)
    data_processing_pipeline(file_path, print_output=True,show_plot=False)


    
 
