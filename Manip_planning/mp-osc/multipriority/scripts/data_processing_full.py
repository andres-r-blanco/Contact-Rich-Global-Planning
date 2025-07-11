import os
import csv
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.figsize': (8, 5)
})

# ADD DIRECTORY PATH HERE
DATA_PATH = "/home/rishabh/Andres/Manip_planning/mp-osc/multipriority/data/manip_data/cost_comparison_reach"


def detect_data_format(filepath):
    with open(filepath, 'r') as f:
        first_line = f.readline()
        second_line = f.readline()
        # print(f"this is the file path {filepath}")
        # print(f"this is the first line {first_line[:50]}...")
        # print(f"this is the second line {second_line[:50]}...")
        if 'Time (s)' in second_line:
            return 'mpc'
        elif 'Trial,1,Sim Type' in first_line:
            return 'planner'
        else:
            raise ValueError("Unknown data format")

def parse_file(filepath):
    data_format = detect_data_format(filepath)
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        data_rows = list(reader)
    data = [item for row in data_rows for item in row]

    trials = {}
    i = 0
    while i < len(data):
        if data[i] == 'Trial':
            trial_num = int(data[i+1])
            trials[trial_num] = {}
            valid_trial = False  # <-- track whether we found manipulability in this trial
            i += 2
            while i < len(data) and data[i] != 'Trial':
                key = data[i]
                values = []
                i += 1

                if data_format == 'planner':
                    if key in ['Closest Taxel Manip', 'Distance to Last Node', 'Total Cost','Local Manip Cost'] or key.startswith('Joint'):
                        if key == 'Closest Taxel Manip':
                            valid_trial = True  # we found a valid manipulability row
                        while i < len(data) and data[i] not in ['Trial', 'Closest Taxel Manip', 'Distance to Last Node', 'Total Cost'] and not data[i].startswith('Joint'):
                            try:
                                values.append(float(data[i]))
                                i += 1
                            except:
                                break
                        if key == 'Distance to Last Node':
                            key = 'Joint Norm Distance'
                            values = np.cumsum(values)
                        elif key.startswith('Joint'):
                            try:
                                joint_num = int(key.split()[1].strip(':')) + 1  # Handle "Joint 0: " format and shift joint numbers up by 1
                                key = f'Joint {joint_num}'
                            except ValueError:
                                raise ValueError(f"Invalid joint number format in key: {key}")
                        trials[trial_num][key] = np.array(values)
                    else:
                        i += 1

                elif data_format == 'mpc':
                    if key in ['Time (s)', 'Joint Norm Distance', 'Local Manip Cost','Closest Taxel Manip', 'Distance to Taxel', 'Closest Taxel ID', 
                               'Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Joint 7']:
                        if key == 'Local Manip Cost':
                            valid_trial = True
                        while i < len(data) and data[i] not in ['Trial', 'Time (s)', 'Joint Norm Distance', 
                                                                'Local Manip Cost', 'Closest Taxel Manip', 
                                                                 'Distance to Taxel', 'Closest Taxel ID', 'Joint 1', 'Joint 2', 
                                                                 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Joint 7']:
                            try:
                                values.append(float(data[i]))
                                i += 1
                            except:
                                break
                        trials[trial_num][key] = np.array(values)
                    else:
                        i += 1

            # Only keep trials that actually had manipulability data
            if not valid_trial:
                del trials[trial_num]
        else:
            i += 1
    return trials, data_format


def ensure_plot_dir(base_path):
    plot_dir = os.path.join(base_path, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

def plot_joint_values_over_distance(trials, base_path, show_plot=False):
    joint_keys = [f'Joint {i}' for i in range(1, 8)]
    joint_plots_dir = os.path.join(base_path, 'plots', 'joint_values_over_distance')
    os.makedirs(joint_plots_dir, exist_ok=True)

    for joint_key in joint_keys:
        plt.figure()
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        has_data = False
        for trial_num, data_dict in trials.items():
            if joint_key in data_dict and len(data_dict[joint_key]) > 0 and 'Joint Norm Distance' in data_dict and len(data_dict['Joint Norm Distance']) > 0:
                cumulative_distance = np.cumsum(data_dict['Joint Norm Distance'])
                plt.plot(cumulative_distance, data_dict[joint_key], label=f'Trial {trial_num}')
                has_data = True
        plt.title(f'{joint_key} Values Over Distance (All Trials)')
        plt.xlabel('Cumulative Joint Norm Distance')
        plt.ylabel(f'{joint_key} Value')
        if has_data:
            plt.legend()
        plt.tight_layout()
        filename = os.path.join(joint_plots_dir, f'{joint_key}_Values_Over_Distance.png')
        plt.savefig(filename)
        if show_plot:
            plt.show()
        plt.close()

def plot_manip_vs_normalized_joint_norm_all(trials, base_path, show_plot=False):
    plt.figure()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    for trial_num, data_dict in trials.items():
        joint_norm = data_dict['Joint Norm Distance']
        manip = data_dict['Closest Taxel Manip']
        cumulative_distance = np.cumsum(joint_norm)
        normalized_distance = cumulative_distance / cumulative_distance[-1]
        plt.plot(normalized_distance, manip, label=f'Trial {trial_num}')
    plt.yscale('log')  # Use log scale for y-axis
    plt.title('Manipulability vs Normalized Joint Norm Distance (All Trials)')
    plt.xlabel('Normalized Cumulative Joint Norm Distance')
    plt.ylabel('Manipulability (Log Scale)')
    plt.legend()
    plt.tight_layout()
    plot_dir = ensure_plot_dir(base_path)
    filename = os.path.join(plot_dir, 'AllTrials_Manip_vs_NormalizedJointNorm.png')
    plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.close()
    
def plot_closest_taxel_vs_normalized_joint_norm(trials, base_path, show_plot=False):
    plt.figure()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    for trial_num, data_dict in trials.items():
        if 'Closest Taxel ID' in data_dict and 'Joint Norm Distance' in data_dict:
            joint_norm = data_dict['Joint Norm Distance']
            closest_taxel = data_dict['Closest Taxel ID']
            cumulative_distance = np.cumsum(joint_norm)
            normalized_distance = cumulative_distance / cumulative_distance[-1]
            plt.plot(normalized_distance, closest_taxel, label=f'Trial {trial_num}')
    plt.title('Closest Taxel vs Normalized Joint Norm Distance (All Trials)')
    plt.xlabel('Normalized Cumulative Joint Norm Distance')
    plt.ylabel('Closest Taxel ID')
    plt.legend()
    plt.tight_layout()
    plot_dir = ensure_plot_dir(base_path)
    filename = os.path.join(plot_dir, 'AllTrials_ClosestTaxel_vs_NormalizedJointNorm.png')
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
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    plt.bar(trial_ids, total_distances)
    plt.title('Total Joint Norm Distance per Trial')
    plt.xlabel('Trial')
    plt.ylabel('Total Joint Norm Distance')
    plt.tight_layout()
    plot_dir = ensure_plot_dir(base_path)
    filename = os.path.join(plot_dir, 'Total_JointNorm_per_Trial.png')
    plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.close()

def plot_manipulability_histogram(trials, base_path, show_plot=False, name=None):
    all_manip = np.concatenate([trials[trial]['Closest Taxel Manip'] for trial in trials])
    plt.figure()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    all_manip = all_manip + 1e-15
    bins = np.logspace(np.log10(1e-15), np.log10(1.01), num=50)
    hist, bin_edges = np.histogram(all_manip, bins=bins)
    percentages = (hist / len(all_manip)) * 100
    plt.bar(bin_edges[:-1], percentages, width=np.diff(bin_edges), edgecolor='black', align='edge')
    plt.xscale('log')
    plt.ylim(0, 80)
    # plt.xlim(1e-15, 9)
    ax.set_xticks([10**i for i in range(-15, 1)])  # Set ticks at every power of 10
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"$10^{{{int(np.log10(x))}}}$"))
    if name is not None:
        plt.suptitle(name, fontsize=9, fontweight='bold')
        plt.title('Histogram of Closest Taxel Manipulability Across All Trials')
    else:
        plt.title('Histogram of Closest Taxel Manipulability Across All Trials')
    plt.xlabel('Closest Taxel Manipulability (Log Scale)')
    plt.ylabel('Percentage (%)')
    plt.tight_layout()
    plot_dir = ensure_plot_dir(base_path)
    filename = os.path.join(plot_dir, 'Closest_Taxel_Manipulability_Histogram.png')
    plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.close()
    
def plot_manip_cost_histogram(trials, base_path, show_plot=False, name=None):
    all_manip = np.concatenate([trials[trial]['Local Manip Cost'] for trial in trials])
    plt.figure()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    bins = np.linspace(0, 1, num=50)  # Use equal-sized bins between 0 and 1
    hist, bin_edges = np.histogram(all_manip, bins=bins)
    percentages = (hist / len(all_manip)) * 100
    plt.bar(bin_edges[:-1], percentages, width=np.diff(bin_edges), edgecolor='black', align='edge')
    plt.ylim(0, 80)
    plt.xlim(0, 1)
    if name is not None:
        plt.suptitle(name, fontsize=9, fontweight='bold')
        plt.title('Histogram of Manipulability Cost Across All Trials')
    else:
        plt.title('Histogram of Manipulability Cost Across All Trials')
    plt.xlabel('Manipulability Cost')
    plt.ylabel('Percentage (%)')
    plt.tight_layout()
    plot_dir = ensure_plot_dir(base_path)
    filename = os.path.join(plot_dir, 'Manip_Cost_Histogram.png')
    plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.close()

def plot_distance_vs_avg_manipulability(trials, base_path, show_plot=False):
    total_distances = []
    avg_manips = []
    for trial_num in trials:
        manip = trials[trial_num]['Closest Taxel Manip']
        joint_norm = trials[trial_num]['Joint Norm Distance']
        total_distance = np.sum(joint_norm)
        avg_manip = np.mean(manip)
        total_distances.append(total_distance)
        avg_manips.append(avg_manip)

    plt.figure()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    plt.scatter(total_distances, avg_manips)
    plt.title('Total Distance vs Average Closest Taxel Manipulability')
    plt.xlabel('Total Joint Norm Distance')
    plt.ylabel('Average Closest Taxel Manipulability')
    plt.tight_layout()
    plot_dir = ensure_plot_dir(base_path)
    filename = os.path.join(plot_dir, 'Distance_vs_AvgManipulability.png')
    plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.close()
    
def plot_distance_vs_manip_cost(trials, base_path, show_plot=False):
    total_distances = []
    avg_manips = []
    for trial_num in trials:
        manip = trials[trial_num]['Local Manip Cost']
        joint_norm = trials[trial_num]['Joint Norm Distance']
        total_distance = np.sum(joint_norm)
        avg_manip = np.mean(manip)
        total_distances.append(total_distance)
        avg_manips.append(avg_manip)

    plt.figure()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    plt.scatter(total_distances, avg_manips)
    plt.title('Total Distance vs Average Manipulability Cost')
    plt.xlabel('Total Joint Norm Distance')
    plt.ylabel('Average Manipulability Cost')
    plt.tight_layout()
    plot_dir = ensure_plot_dir(base_path)
    filename = os.path.join(plot_dir, 'Distance_vs_AvgManipCost.png')
    plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.close()

def plot_distance_vs_low_manip_states(trials, base_path, show_plot=False):
    total_distances = []
    low_manip_states = []
    for trial_num in trials:
        manip = trials[trial_num]['Closest Taxel Manip']
        joint_norm = trials[trial_num]['Joint Norm Distance']
        total_distance = np.sum(joint_norm)
        num_low_manip_states = np.sum(manip < 0.01)
        total_distances.append(total_distance)
        low_manip_states.append(num_low_manip_states)

    plt.figure()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    plt.scatter(total_distances, low_manip_states)
    plt.title('Total Distance vs Number of States with Manipulability < 0.01')
    plt.xlabel('Total Joint Norm Distance')
    plt.ylabel('Number of States with Manipulability < 0.01')
    plt.tight_layout()
    plot_dir = ensure_plot_dir(base_path)
    filename = os.path.join(plot_dir, 'Distance_vs_LowManipStates.png')
    plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.close()

def plot_manipulability_boxplot(trials, base_path, show_plot=False):
    data = [trials[trial]['Closest Taxel Manip'] for trial in trials]
    labels = [f'Trial {trial}' for trial in trials]
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title('Boxplot of Closest Taxel Manipulability Across Trials')
    plt.xlabel('Trial')
    plt.ylabel('Closest Taxel Manip')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_dir = ensure_plot_dir(base_path)
    filename = os.path.join(plot_dir, 'Closest Taxel Manipulability_Boxplot.png')
    plt.savefig(filename, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()

def compute_trial_metrics(trials, trial_num, data_format):
    data = trials[trial_num]
    manip = data['Closest Taxel Manip']
    manip_cost = data['Local Manip Cost']
    # print(f"mean manip: {np.mean(manip)}")
    joint_norm = data['Joint Norm Distance']
    avg_manip = np.mean(manip)
    avg_manip_cost = np.mean(manip_cost)
    num_low_manip = np.sum(manip < 0.01)
    num_high_manip_cost = np.sum(manip_cost > 0.35)
    total_distance = np.sum(joint_norm)
    total_time = len(joint_norm) if data_format == 'planner' else (data['Time (s)'][-1] - data['Time (s)'][0])
    return avg_manip, num_low_manip, total_distance, total_time, avg_manip_cost, num_high_manip_cost

def compute_average_metrics_across_trials(trials, data_format, base_path, print_output=False):
    avg_manips = []
    avg_manip_costs = []
    num_low_manips = []
    num_high_manip_costs = []
    total_distances = []
    total_times = []
    percentages_low_manip = []
    percentage_high_manip_costs = []
    print(f"Number of trials: {len(trials)}")
    for trial_num in trials:
        avg_manip, num_low_manip, total_distance, total_time, avg_manip_cost, num_high_manip_cost= compute_trial_metrics(trials, trial_num, data_format)
        avg_manips.append(avg_manip)
        avg_manip_costs.append(avg_manip_cost)
        num_low_manips.append(num_low_manip)
        num_high_manip_costs.append(num_high_manip_cost)
        total_distances.append(total_distance)
        total_times.append(total_time)
        percentage_low_manip = (num_low_manip / len(trials[trial_num]['Closest Taxel Manip'])) * 100 #TODO this is wrong
        percentages_low_manip.append(percentage_low_manip)
        percentage_high_manip_costs = (num_high_manip_cost / len(trials[trial_num]['Local Manip Cost'])) * 100

    mean_avg_manip = np.mean(avg_manips)
    mean_avg_manip_cost = np.mean(avg_manip_costs)
    mean_num_low_manip = np.mean(num_low_manips)
    mean_num_high_manip_cost = np.mean(num_high_manip_costs)
    mean_total_distance = np.mean(total_distances)
    mean_total_time = np.mean(total_times)
    mean_percentage_low_manip = np.mean(percentages_low_manip)
    mean_percentage_high_manip_costs = np.mean(percentage_high_manip_costs)

    std_avg_manip = np.std(avg_manips)
    std_avg_manip_cost = np.std(avg_manip_costs)
    std_num_low_manip = np.std(num_low_manips)
    std_num_high_manip_cost = np.std(num_high_manip_costs)
    std_total_distance = np.std(total_distances)
    std_total_time = np.std(total_times)
    std_percentage_low_manip = np.std(percentages_low_manip)
    std_percentage_high_manip_costs = np.std(percentage_high_manip_costs)

    metrics_file = os.path.join(base_path, "average_metrics.csv")
    os.makedirs(base_path, exist_ok=True)
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Mean", "Standard Deviation"])
        writer.writerow(["Average Closest Taxel Manipulability", mean_avg_manip, std_avg_manip])
        writer.writerow(["Number of states with closest taxel manipulability < 0.01", mean_num_low_manip, std_num_low_manip])
        writer.writerow(["Percentage of states with closest taxel manipulability < 0.01", mean_percentage_low_manip, std_percentage_low_manip])
        writer.writerow(["Average Manipulability Cost", mean_avg_manip_cost, std_avg_manip_cost])
        writer.writerow(["Number of states with manipulability cost > 0.35", mean_num_high_manip_cost, std_num_high_manip_cost])
        writer.writerow(["Percentage of states with manipulability cost > 0.35", mean_percentage_high_manip_costs, std_percentage_high_manip_costs])
        writer.writerow(["Total Joint Norm Distance", mean_total_distance, std_total_distance])
        writer.writerow(["Total Time", mean_total_time, std_total_time])

    if print_output:
        print("\nAverage metrics across all trials:")
        print(f"Average Closest Taxel Manipulability: {mean_avg_manip:.4f} ± {std_avg_manip:.4f}")
        print(f"Number of states with closest taxel manipulability < 0.01: {mean_num_low_manip:.2f} ± {std_num_low_manip:.2f}")
        print(f"Percentage of states with closest taxel manipulability < 0.01: {mean_percentage_low_manip:.2f}% ± {std_percentage_low_manip:.2f}%")
        print(f"Average Manipulability Cost: {mean_avg_manip_cost:.4f} ± {std_avg_manip_cost:.4f}")
        print(f"Number of states with manipulability cost > 0.35: {mean_num_high_manip_cost:.2f} ± {std_num_high_manip_cost:.2f}")
        print(f"Percentage of states with manipulability cost > 0.35: {mean_percentage_high_manip_costs:.2f}% ± {std_percentage_high_manip_costs:.2f}%")
        print(f"Total Joint Norm Distance: {mean_total_distance:.4f} ± {std_total_distance:.4f}")
        print(f"Total Time: {mean_total_time:.4f} ± {std_total_time:.4f}")

    return mean_avg_manip, std_avg_manip, mean_total_distance,std_total_distance, mean_avg_manip_cost, std_avg_manip_cost, mean_percentage_high_manip_costs, std_percentage_high_manip_costs

# def data_processing_pipeline(file_path, print_output=False, show_plot=False):
#     trials, data_format = parse_file(file_path)
#     base_path = file_path.rsplit('.', 1)[0]
#     name = os.path.basename(file_path).rsplit('.', 1)[0]
#     compute_average_metrics_across_trials(trials, data_format, base_path, print_output)
#     plot_manip_vs_normalized_joint_norm_all(trials, base_path, show_plot)
#     plot_total_joint_norm_per_trial(trials, base_path, show_plot)
#     plot_manipulability_histogram(trials, base_path, show_plot, name = name)
#     plot_distance_vs_avg_manipulability(trials, base_path, show_plot)
#     plot_joint_values_over_distance(trials, base_path, show_plot)
#     plot_distance_vs_low_manip_states(trials, base_path, show_plot)
#     plot_manipulability_boxplot(trials, base_path, show_plot)

def batch_process_folder(folder_path, print_output=False, show_plot=False):
    summary_file = os.path.join(folder_path, "summary_metrics.csv")
    with open(summary_file, "w", newline="") as f_summary:
        writer = csv.writer(f_summary)
        writer.writerow(["File", "Mean Avg Manip", "Std Avg Manip", "Mean Total Distance", "Std Total Distance", "Mean Avg Manip Cost", "Std Avg Manip Cost", "Mean % High Manip Cost", "Std % High Manip Cost"])

        for filename in os.listdir(folder_path):
            if filename.endswith(".csv") and filename != "summary_metrics.csv":
                file_path = os.path.join(folder_path, filename)
                name = os.path.basename(file_path).rsplit('.', 1)[0]
                print(f"\nProcessing {filename}")
                trials, data_format = parse_file(file_path)
                base_path = file_path.rsplit('.', 1)[0]
                mean_avg_manip, std_avg_manip, mean_total_distance,std_total_distance, mean_avg_manip_cost, std_avg_manip_cost, mean_percentage_high_manip_costs, std_percentage_high_manip_costs = compute_average_metrics_across_trials(trials, data_format, base_path, print_output)
                writer.writerow([filename, mean_avg_manip, std_avg_manip, mean_total_distance,std_total_distance, mean_avg_manip_cost, std_avg_manip_cost, mean_percentage_high_manip_costs, std_percentage_high_manip_costs])
                plot_manip_vs_normalized_joint_norm_all(trials, base_path, show_plot)
                plot_closest_taxel_vs_normalized_joint_norm(trials, base_path, show_plot)
                plot_total_joint_norm_per_trial(trials, base_path, show_plot)
                plot_manipulability_histogram(trials, base_path, show_plot,name=name)
                plot_distance_vs_avg_manipulability(trials, base_path, show_plot)
                plot_distance_vs_manip_cost(trials, base_path, show_plot)
                plot_manip_cost_histogram(trials, base_path, show_plot, name=name)
                plot_distance_vs_low_manip_states(trials, base_path, show_plot)
                plot_joint_values_over_distance(trials, base_path, show_plot)
                plot_manipulability_boxplot(trials, base_path, show_plot)

def compare_manipulability_across_files(folder_path):
    file_manip_map = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv") and filename != "summary_metrics.csv":
            file_path = os.path.join(folder_path, filename)
            trials, _ = parse_file(file_path)
            # Concatenate all manipulability arrays in this file
            all_manip = np.concatenate([trials[t]['Closest Taxel Manip'] for t in trials])
            file_manip_map[filename] = all_manip

    files = list(file_manip_map.keys())
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            if np.array_equal(file_manip_map[files[i]], file_manip_map[files[j]]):
                print(f"Manipulability arrays are IDENTICAL between {files[i]} and {files[j]}")
            else:
                return# print(f"Manipulability arrays are DIFFERENT between {files[i]} and {files[j]}")

def add_csv_extension_to_files(folder_path):
    for filename in os.listdir(folder_path):
        if not filename.endswith(".csv"):
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, f"{filename}.csv")
            os.rename(old_path, new_path)


def plot_summary_metrics(folder_path, show_plots=False):
    summary_file = os.path.join(folder_path, "summary_metrics.csv")
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Summary file not found at {summary_file}")

    weights = []
    contact_sample_chances = []
    obj_reductions = []
    avg_manipulabilities = []
    avg_manip_costs = []
    avg_total_distances = []
    avg_percentage_high_manip_costs = []
    filenames = []

    with open(summary_file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            filename, mean_avg_manip, std_avg_manip, mean_total_distance,std_total_distance, mean_avg_manip_cost, std_avg_manip_cost, mean_percentage_high_manip_costs, std_percentage_high_manip_costs = row
            filenames.append(filename)
            if "weight" in filename and "contactsamplechance" in filename and "objreduction" in filename:
                weight = float(filename.split("weight")[1].split("_")[0])
                contact_sample_chance = float(filename.split("contactsamplechance")[1].split("_")[0])
                obj_reduction = float(filename.split("objreduction")[1].split("_")[0])
                weights.append(weight)
                contact_sample_chances.append(contact_sample_chance)
                obj_reductions.append(obj_reduction)
                avg_manipulabilities.append(float(mean_avg_manip))
                avg_manip_costs.append(float(mean_avg_manip_cost))
                avg_total_distances.append(float(mean_total_distance))
                avg_percentage_high_manip_costs.append(float(mean_percentage_high_manip_costs))

    # Plot metrics vs weight
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(False)
    ax1.set_xlabel("Weight")
    ax1.set_ylabel("Avg Total Distance", color="red")
    ax1.scatter(weights, avg_total_distances, label="Avg Total Distance", color="red", marker="x")
    ax1.tick_params(axis="y", labelcolor="red")

    ax2 = ax1.twinx()  # Create a second y-axis
    ax2.set_ylabel("Average Closest Taxel Manipulability", color="blue")
    ax2.scatter(weights, avg_manipulabilities, label="Average Closest Taxel Manipulability", color="blue", marker="o")
    ax2.tick_params(axis="y", labelcolor="blue")

    fig.tight_layout()
    plt.title("Average Metrics vs Weight")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "Summary_Metrics_vs_Weight.png"))
    if show_plots:
        plt.show()

    # Plot metrics vs object reduction
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(False)
    ax1.set_xlabel("Object Reduction")
    ax1.set_ylabel("Avg Total Distance", color="red")
    ax1.scatter(obj_reductions, avg_total_distances, label="Avg Total Distance", color="red", marker="x")
    ax1.tick_params(axis="y", labelcolor="red")

    ax2 = ax1.twinx()  # Create a second y-axis
    ax2.set_ylabel("Average Closest Taxel Manipulability", color="blue")
    ax2.scatter(obj_reductions, avg_manipulabilities, label="Average Closest Taxel Manipulability", color="blue", marker="o")
    ax2.tick_params(axis="y", labelcolor="blue")

    fig.tight_layout()
    plt.title("Average Metrics vs Object Reduction")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "Summary_Metrics_vs_ObjectReduction.png"))
    if show_plots:
        plt.show()
        
    
    old_cost_runs = []
    new_cost_runs = []
    for i, filename in enumerate(filenames):
        if "old_costTrue" in filename and "old_closestTrue" in filename:
            old_cost_runs.append((avg_total_distances[i], avg_manipulabilities[i]))
        elif "old_closestTrue" in filename:
            new_cost_runs.append((avg_total_distances[i], avg_manipulabilities[i]))

    old_cost_runs = np.array(old_cost_runs)
    new_cost_runs = np.array(new_cost_runs)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(False)
    ax1.set_xlabel("Average Total Distance")
    ax1.set_ylabel("Average Closest Taxel Manipulability", color="blue")

    if len(old_cost_runs) > 0:
        ax1.scatter(old_cost_runs[:, 0], old_cost_runs[:, 1], label="Old Cost", color="blue", marker="o")
    if len(new_cost_runs) > 0:
        ax1.scatter(new_cost_runs[:, 0], new_cost_runs[:, 1], label="New Cost", color="red", marker="x")

    ax1.tick_params(axis="y", labelcolor="blue")
    plt.legend()
    plt.title("Average Metrics: Old Cost vs New Cost (Filtered by old_closestFalse)")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "Summary_Metrics_OldCost_vs_NewCost_Filtered.png"))
    if show_plots:
        plt.show()
        
        
    old_closest_runs = []
    new_closest_runs = []
    for i, filename in enumerate(filenames):
        if "old_costFalse" in filename:  # Only consider runs with "old_costTrue" in the filename
            if "old_closestTrue" in filename:
                old_closest_runs.append((avg_total_distances[i], avg_manipulabilities[i]))
            else:
                new_closest_runs.append((avg_total_distances[i], avg_manipulabilities[i]))

    old_closest_runs = np.array(old_closest_runs)
    new_closest_runs = np.array(new_closest_runs)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(False)
    ax1.set_xlabel("Average Total Distance")
    ax1.set_ylabel("Average Closest Taxel Manipulability", color="blue")

    if len(old_closest_runs) > 0:
        ax1.scatter(old_closest_runs[:, 0], old_closest_runs[:, 1], label="Old Closest", color="blue", marker="o")
    if len(new_closest_runs) > 0:
        ax1.scatter(new_closest_runs[:, 0], new_closest_runs[:, 1], label="New Closest", color="red", marker="x")

    ax1.tick_params(axis="y", labelcolor="blue")
    plt.legend()
    plt.title("Average Metrics: Old Closest vs New Closest (Filtered by old_costTrue)")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "Summary_Metrics_OldClosest_vs_NewClosest_Filtered.png"))
    if show_plots:
        plt.show()

    # Plot metrics vs contact sample chance
    # fig, ax1 = plt.subplots(figsize=(10, 6))
    # ax1.spines['top'].set_visible(False)
    # ax1.spines['right'].set_visible(False)
    # ax1.grid(False)
    # ax1.set_xlabel("Contact Sample Chance")
    # ax1.set_ylabel("Avg Total Distance", color="red")
    # ax1.scatter(contact_sample_chances, avg_total_distances, label="Avg Total Distance", color="red", marker="x")
    # ax1.tick_params(axis="y", labelcolor="red")

    # ax2 = ax1.twinx()  # Create a second y-axis
    # ax2.set_ylabel("Number of States Under 0.01", color="blue")
    # ax2.scatter(contact_sample_chances, avg_manipulabilities, label="Number of States Under 0.01", color="blue", marker="o")
    # ax2.tick_params(axis="y", labelcolor="blue")

    # fig.tight_layout()
    # plt.title("Average Metrics vs Contact Sample Chance")
    # plt.tight_layout()
    # plt.savefig(os.path.join(folder_path, "Summary_Metrics_vs_ContactSampleChance.png"))
    # if show_plots:
    #     plt.show()

    # Plot manipulability vs total distance for all runs
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    for i in range(len(avg_manipulabilities)):
        plt.scatter(avg_total_distances[i], avg_manipulabilities[i], label=filenames[i])
    plt.title("Manipulability vs Total Distance (All Runs)")
    plt.xlabel("Average Total Distance")
    plt.ylabel("Average Closest Taxel Manipulability")
    plt.grid()
    plt.tight_layout()
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize="small", ncol=3)
    plt.savefig(os.path.join(folder_path, "Manipulability_vs_TotalDistance_AllRuns.png"), bbox_inches="tight")
    if show_plots:
        plt.show()
        
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    for i in range(len(avg_manipulabilities)):
        plt.scatter(avg_total_distances[i], avg_manip_costs[i], label=filenames[i])
    plt.title("Manipulability Cost vs Total Distance (All Runs)")
    plt.xlabel("Average Total Distance")
    plt.ylabel("Average Manipulability Cost")
    plt.grid()
    plt.tight_layout()
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize="small", ncol=3)
    plt.savefig(os.path.join(folder_path, "Manipulability_Cost_vs_TotalDistance_AllRuns.png"), bbox_inches="tight")
    if show_plots:
        plt.show()
        
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    for i in range(len(avg_manipulabilities)):
        plt.scatter(avg_total_distances[i], avg_percentage_high_manip_costs[i], label=filenames[i])
    plt.title("Percentage of States Above 0.35 Manipulability Cost vs Total Distance (All Runs)")
    plt.ylabel("Percentage of States Above 0.35 Manipulability Cost")
    plt.xlabel("Average Total Distance")
    plt.grid()
    plt.tight_layout()
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize="small", ncol=3)
    plt.savefig(os.path.join(folder_path, "Percentage_States_Manipulability_Cost_vs_TotalDistance_AllRuns.png"), bbox_inches="tight")
    if show_plots:
        plt.show()

if __name__ == "__main__":                  
    # add_csv_extension_to_files(DATA_PATH)
    compare_manipulability_across_files(DATA_PATH)
    batch_process_folder(DATA_PATH, print_output=True, show_plot=False)
    plot_summary_metrics(DATA_PATH)
