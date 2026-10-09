import glob
import re


#"hyper_FeatureNoise"
data_name = "hyper_homo02"
tune_name = "FPLPGCN_MLP_" + data_name + "_tune*"

if data_name == "cora" or data_name == "citeseer":
    tune_name = data_name + "_tune*"
if data_name.startswith("hyper_homo"):
    tune_name = data_name + "_tune*"

# sbatch_file path name
sbatch_file_name = "sbatch_script/tune_scripts/"+ data_name +'_tune.sbatch'
optimal_setting = "optimal_settings/optimal_setting_" + data_name + "_MLP.out"

max_number = None
max_file = None

# Get a list of all files on the dataset
tune_files = glob.glob("output/" + tune_name)

# find the model that report best validation score
pattern = r"Val Average Precision Score: (\d+\.\d+)"
if data_name == "cora" or data_name == "citeseer":
    pattern = r"Val acc (\d+\.\d+)"

for filename in tune_files:
    with open(filename, 'r') as file:
        last_line = list(file)[-1]
        match = re.search(pattern, last_line)
        if match:
            score = float(match.group(1))
            print(f"File: {file}, Score: {score}")
            if score is not None and (max_number is None or score > max_number):
                max_number = score
                max_file = filename
        else:
            print(f"did not found the Val Ave in File: {file}")

# Extract the number from the filename
print(max_file)
number = re.findall(r'\d+', max_file)
index = int(number[0]) if number else None

print(index, number)
# Open 'tune.sbatch' and find the line with the corresponding index
if index is not None:
    with open(sbatch_file_name, 'r') as file:
        lines = file.readlines()
        args_lines = [line for line in lines if 'ARGS' in line]
    # Find and print the line that starts with the index
    for line in args_lines:
        if line.startswith("    "+str(index)+")"):
            print("the optimal setting: ")
            print(line)
            with open(optimal_setting, 'w') as file:
                file.write(line)


