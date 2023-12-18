import csv
from faker import Faker
import random
import string

def generate_fake_student_data(num_rows):
    # Initialize Faker for generating fake data
    fake = Faker()

    # Initialize the data list with column headers
    data = [('StudentID', 'Marks', 'IQ')]

    # Set to keep track of generated student IDs to ensure uniqueness
    generated_ids = set()

    # Generate fake data for the specified number of rows
    for _ in range(num_rows):
        # Generate a unique student ID
        student_id = generate_unique_id(generated_ids)

        # Generate random marks and IQ for the student
        marks, iq = generate_student_attributes(fake)

        # Append the student data to the list
        data.append((student_id, marks, iq))

    return data

def generate_unique_id(existing_ids):
    # Generate a unique student ID that is not in the set of existing IDs
    while True:
        student_id = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        if student_id not in existing_ids:
            existing_ids.add(student_id)
            return student_id

def generate_student_attributes(fake):
    # Generate a random IQ between 80 and 150
    iq = fake.random_int(min=80, max=150)

    # Calculate the probability of having high marks based on IQ
    marks_probability = min(1.0, iq / 150.0)

    # Introduce a random factor to the probability
    random_factor = random.uniform(0.8, 1.2)
    marks_probability *= random_factor

    # Generate a random value to determine if the mark is systematic or random
    mark_type_prob = random.random()

    # Determine marks based on probability
    if mark_type_prob < marks_probability:
        marks = fake.random_int(min=10, max=20)  # High Marks
    else:
        marks = fake.random_int(min=0, max=20)   # Random Marks

    return marks, iq

def write_to_csv(file_name, data):
    # Write the generated data to a CSV file
    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(data)

if __name__ == "__main__":
    # Adjust the number of rows as needed
    num_rows = 500

    # Generate fake student data
    fake_student_data = generate_fake_student_data(num_rows)

    # Write data to CSV file
    csv_file_name = 'csvExamples/randomStudentData.csv'
    write_to_csv(csv_file_name, fake_student_data)

    print(f"Random CSV file '{csv_file_name}' with {num_rows} rows generated.")
