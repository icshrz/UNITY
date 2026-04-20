import re
import json

# Define a function to extract the desired data from a log file
def extract_log_data(file_name):
    with open(file_name, 'r', encoding='ISO-8859-1') as file:
        content = file.read()

    # Regular expressions for matching the required data
    node_similarity_pattern = r"node similarity:\s*([\d\.]+)"
    edge_similarity_pattern = r"edge similarity:\s*([\d\.]+)"
    nid_info_pattern = r"nid shape:\s*(\d+)\s*eid shape:\s*(\d+)\s*nid no repeat:\s*(\d+)\s*eid no repeat:\s*(\d+)"
    
    # Extract the data
    node_similarity = re.findall(node_similarity_pattern, content)
    edge_similarity = re.findall(edge_similarity_pattern, content)
    nid_info = re.findall(nid_info_pattern, content)

    # If there are no matches for any of these values, return an empty list
    if not (node_similarity and edge_similarity and nid_info):
        print(f"No matching data found in file {file_name}.")
        return []

    # Calculate the ratios for nid and eid (1 - ratio)
    nid_ratios = [1 - (int(nid_repeat)/int(nid_shape)) for nid_shape, _, nid_repeat, _ in nid_info]
    eid_ratios = [1 - (int(eid_repeat)/int(eid_shape)) for _, eid_shape, _, eid_repeat in nid_info]

    # Combine the extracted data into a sequence of dictionaries
    result = []
    for ns, es, nid, eid in zip(node_similarity, edge_similarity, nid_ratios, eid_ratios):
        result.append({
            'Node Similarity': ns,
            'Edge Similarity': es,
            'NID Ratio': nid,
            'EID Ratio': eid
        })

    return result

# Define a function to save the extracted data to a JSON file
def save_extracted_data_to_json(file_name, data):
    output_file_name = file_name.replace('demo', 'extracted_') + '.json'
    with open(output_file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Files to extract data from
files = ['demoCA', 'demoGD', 'demoRED', 'demoWIKI']

# Extract data from each file and save to a new JSON file
for file_name in files:
    extracted_data = extract_log_data(file_name)
    if extracted_data:  # Only save data if extraction was successful
        save_extracted_data_to_json(file_name, extracted_data)

"Files have been saved successfully as JSON."
