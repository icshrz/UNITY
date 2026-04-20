import yaml

# Function to recursively multiply specific keys by 1.1 and round to one decimal place
def multiply_values(data):
    # If the current data is a dictionary, apply the function recursively
    if isinstance(data, dict):
        for key, value in data.items():
            if key in ['Sample', 'CPU Updater', 'GPU Updater', 'Linear', 'Attention']:
                data[key] = round(value * 1.1, 1)  # Multiply the value by 1.1 and round to 1 decimal place
            else:
                multiply_values(value)  # Recurse into the next dictionary
    elif isinstance(data, list):  # If the current data is a list, iterate through each item
        for item in data:
            multiply_values(item)

# Load the YAML data from the file, ensuring the order is preserved
with open('total2.yaml', 'r') as file:
    data = yaml.safe_load(file)

# Modify the relevant fields in the data
multiply_values(data)

# Save the modified data back to a new YAML file, ensuring the order is preserved
with open('modified_total2.yaml', 'w') as file:
    yaml.dump(data, file, default_flow_style=False, allow_unicode=True, sort_keys=False)

print("Data has been modified and saved to 'modified_total2.yaml'.")
