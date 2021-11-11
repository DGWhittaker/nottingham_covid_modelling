import csv


def write_to_csv(csv_data, csv_file, csv_columns):
    with open(csv_file, 'w', newline='\n') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in csv_data:
            writer.writerow(data)


def read_csv(cvs_file):
    file = open(cvs_file, newline='')
    reader = csv.reader(file)
    data = list(reader)
    file.close()
    return data
