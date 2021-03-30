import extract_data
import CNN_data

file = "data_set/MIN 30000110701346.txt"
data = extract_data.extract_data_from_txt(file)
label_file = "data_set/30000110701346_label.txt"
label = extract_data.extract_label_from_txt(label_file)
input_data = data.Value.values.astype(int)
print(input_data)

samples = CNN_data.create_samples(input_data, 2016)
print(samples)
