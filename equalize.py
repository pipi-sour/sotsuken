import os
from operator import itemgetter

# number of processes
size = 3
# number of pages
page_size = 10
# modulo
mod = page_size % size
# directory
img_dir = 'pdf2img/'

file_data = []
for i in range(0, page_size):
    file_size = os.path.getsize('{}ouyou{}.png'.format(img_dir, i))
    file_data.append({'name': "test" + str(i), 'size': round(file_size, 2)})
file_data_s = sorted(file_data, key=itemgetter('size'), reverse=True)
# print(size_srtd)

# jobs each workers
jobs = []
# create 2d list for jobs
for i in range(0, size):
    jobs.append([])
# sizes of images
worker_size = [0] * size

for i in range(0, page_size):
    min_size_index = worker_size.index(min(worker_size))
    jobs[min_size_index].append(file_data_s[i])
    worker_size[min_size_index] += file_data_s[i].get("size")
for i in range(0, size):
    print("{}\nlength of {}: {}".format(jobs[i], i, len(jobs[i])))
print(worker_size)
