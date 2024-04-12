import re
import csv
import os 
import numpy as np 

# Function to extract data from a line
def extract_data(line):
    # Regular expression pattern to extract data
    pattern = r'Location\(x=(.*?), y=(.*?), z=(.*?)\), Rotation\(pitch=(.*?), yaw=(.*?), roll=(.*?)\)'
    match = re.search(pattern, line)
    if match:
        x, y, z, pitch, yaw, roll = match.groups()
        return [x, y, z, roll, pitch, yaw]
    else:
        return None

def write_gt_traj(input_file,output_file):
    # Read data from the text file and write to CSV
    with open(input_file, 'r') as file:
        with open(output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            #csv_writer.writerow(['x', 'y', 'z', 'roll', 'pitch', 'yaw'])  # Write header
            for line in file:
                #print("line: ",line)
                data = extract_data(line)
                #print("data: ",data)
                if data:
                    csv_writer.writerow(data)

def even_rate_sampling(array, k):
    n = array.shape[0]
    step = n / (k - 1)  # Calculate the step size
    #print("n: {}, step: {}".format(n, step))
    sampled_indices = [min(int(i * step), n-1) for i in range(k)]  # Sample at even intervals
    #print("len(sampled_indices):", len(sampled_indices))
    if len(sampled_indices) != k:
        raise OSError
    return np.array([array[i] for i in sampled_indices])

def get_gt_car_data(input_file,sim_length): 
    output_file = input_file[:-3] + "csv"
    if not os.path.exists(output_file): 
        write_gt_traj(input_file,output_file)
    data = np.genfromtxt(output_file,delimiter=",")

    downsampled_data = even_rate_sampling(data,sim_length)
    if not downsampled_data.shape[0] == sim_length:
        print("downsampled_data.shape: ",downsampled_data.shape)
        raise OSError
    return downsampled_data