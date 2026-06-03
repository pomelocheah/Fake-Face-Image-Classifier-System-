import os

real_path = "data/archive-2/real"
fake_path = "data/archive-2/fake"

real_count = len(os.listdir(real_path))
fake_count = len(os.listdir(fake_path))

print("Real:", real_count)
print("Fake:", fake_count)