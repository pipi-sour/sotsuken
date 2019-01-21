""" Tesseract OCR Test Program with MPI """
# coding: UTF-8

import sys
import os
from operator import itemgetter
from PIL import Image
from mpi4py import MPI
from pdf2image import convert_from_path
import regex
import pyocr
import pyocr.builders
import csv

PDF_NAME = str(sys.argv[1])
PDF_DIR = 'pdf/'
IMG_DIR = 'pdf2img/' + PDF_NAME + '/'
IMG_TYPE = '.png'
TXT_DIR = 'txt/'
BENCHMARK_LOC = 'bm/test.png'

MASTER_NODE = 0


def create_2dlist(num):
    array = []
    for _ in range(num):
        array.append([])
    return array


def ocr_tool_select():
    tools = pyocr.get_available_tools()
    if not tools:
        print('[{}{}]No OCR tool found.'.format(rank, proc_name))
        sys.exit(1)
    my_tool = tools[0]
    # print('[{}]Will use tool {}'.format(rk, my_tool.get_name()))
    # langs = my_tool.get_available_languages()
    # print("Available languages: %s" % ", ".join(langs))
    my_lang = 'jpn'
    # print("[{}]Will use lang {}".format(rk, my_lang))
    return my_tool, my_lang


def image_analyze(jn, ocr_cfg):
    analyze_start = MPI.Wtime()
    path = IMG_DIR + str(jn) + IMG_TYPE
    print('[{}]Analyzing {}...'.format(rank, path))
    raw_text = ocr_cfg[0].image_to_string(
        Image.open(path),
        lang=ocr_cfg[1],
        builder=pyocr.builders.TextBuilder(tesseract_layout=6)
    )
    raw_text2 = regex.sub(r'((\s)*((\p{Han})|(\p{Hiragana})|(\p{Katakana})|[、。？！])|(\n){2})', r'\3', raw_text)
    text = regex.sub(r'\n|([ ]+)', r' ', raw_text2)
    # print("[{}]Analyzing finished!: {}".format(rk, text))
    analyze_time = MPI.Wtime() - analyze_start
    print("[{}]Analyzing finished!".format(rank))
    print("Time[%s]: %4.5fs." % (jn, analyze_time))
    return text, analyze_time


def benchmark(ocr_cfg):
    bm_start = MPI.Wtime()
    ocr_cfg[0].image_to_string(
        Image.open(BENCHMARK_LOC),
        lang=ocr_cfg[1],
        builder=pyocr.builders.TextBuilder(tesseract_layout=6)
    )
    bm_time = MPI.Wtime() - bm_start
    return bm_time


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
proc_name = MPI.Get_processor_name()
print("Running from processor %s, rank %d out of %d processors." % (proc_name, rank, size))
comm.Barrier()

if rank == MASTER_NODE:
    whole_start = MPI.Wtime()

ocr_config = ocr_tool_select()
bm_result = benchmark(ocr_config)
bm_results = comm.gather(bm_result, root=0)

comm.Barrier()

if rank == MASTER_NODE:
    PAGE_NUM = 0

    print(bm_results)
    if not os.path.exists(IMG_DIR):
        os.mkdir(IMG_DIR)
    pdf_images = convert_from_path('{0}{1}.pdf'.format(PDF_DIR, PDF_NAME))
    for image in pdf_images:
        image.save('{0}{1}{2}{3}'.format(IMG_DIR, PDF_NAME, PAGE_NUM, IMG_TYPE))
        PAGE_NUM += 1

    file_data = []
    # print('Number of Pages: ' + str(PAGE_NUM))

    # get size of files and store them
    for i in range(PAGE_NUM):
        file_size = os.path.getsize('{}{}{}{}'.format(IMG_DIR, PDF_NAME, i, IMG_TYPE))
        file_data.append({'name': PDF_NAME + str(i), 'size': round(file_size, 2)})
    # sort by filesize ascending
    file_data_s = sorted(file_data, key=itemgetter('size'), reverse=True)

    jobs = create_2dlist(size)
    # sizes of images
    worker_size = [0] * size

    for i in range(rank):
        bm_results[i] /= bm_results[0]

    # job equalizing
    for i in range(PAGE_NUM):
        min_size_index = worker_size.index(min(worker_size))
        jobs[min_size_index].append(file_data_s[i])
        worker_size[min_size_index] += file_data_s[i].get("size") * bm_results[min_size_index]
    # for i in range(size):
    #    print("[{}]{}\tlength: {}({})".format(i, jobs[i], len(jobs[i]), worker_size[i]))

    req = create_2dlist(size)
    # Send workers
    for i in range(1, size):
        # print('Now sending to ' + str(i))
        comm.send(len(jobs[i]), dest=i, tag=i)
        # print('jobs[{}]: {}'.format(i, jobs[i]))
        for j in range(len(jobs[i])):
            msg = jobs[i][j]['name']
            req[i].append(comm.isend(msg, dest=i, tag=i))
            # print('MSG to {}: {}, {}'.format(i, msg, req[i][j]))
            req[i][j].wait()
        # print('[%d]All Sent to No.%d!' % (rank, i))

    my_jobs = []
    my_num = len(jobs[0])
    for i in range(0, my_num):
        my_jobs.append(jobs[0][i]['name'])
    print('Jobs of master: {}'.format(my_jobs))

    ocr_config = ocr_tool_select()
    master_result = []
    proc_start = MPI.Wtime()
    for job_name in my_jobs:
        analyze_res = image_analyze(job_name, ocr_config)
        master_result.append({
            'name': int(job_name.replace(PDF_NAME, '')),
            'text': analyze_res[0],
            'time': analyze_res[1]
        })
    print('[0]Task completed!: %5.4fs' % (MPI.Wtime() - proc_start))

    # Receive from workers
    for i in range(1, size):
        # print("Now receiving from {}".format(i))
        req = comm.irecv(source=i, tag=0)
        recv_result = req.wait()
        # print("[0]Received!!".format(i))
        # print('[{}]JOB_NUM: {}'.format(i, len(recv_result)))
        for j in range(len(recv_result)):
            master_result.append(recv_result[j])

    # print('All received!')
    result_s = sorted(master_result, key=itemgetter('name'))

    f = open(TXT_DIR + PDF_NAME + '.txt', 'w')
    f_csv = open(TXT_DIR + PDF_NAME + '.csv', 'w')
    writer = csv.writer(f_csv, lineterminator='\n')
    writer.writerow(['name', 'size[kB]', 'time[s]'])
    for i in range(PAGE_NUM):
        result_s[i]['size'] = file_data[i]['size']
        result_s[i]['text'].encode('utf-8')
        f.write('{}{}{}:\n\n{}\n\n\n'.format(PDF_NAME, i, IMG_TYPE, result_s[i]['text']))
        writer.writerow([PDF_NAME + str(result_s[i]['name']), float(result_s[i]['size']/1000), result_s[i]['time']])

    f.close()
    f_csv.close()
    # print('Writing Complete!\nLocation:{}'.format(TXT_DIR))
    print(u"Congratulations!: %5.4fs." % (MPI.Wtime() - whole_start))

if rank != MASTER_NODE:
    my_jobs = []
    req = comm.irecv(source=0, tag=rank)
    my_num = req.wait()
    req = []
    print('[{}] of job: {}\nReceiving...'.format(rank, my_num))
    tmp_res = []
    for i in range(0, my_num):
        req.append(comm.irecv(source=0, tag=rank))
        my_jobs.append(req[i].wait())
        # print(my_jobs)
    # print('[{}]Received!'.format(rank))
    # print(my_jobs)

    ocr_config = ocr_tool_select()
    my_result = []
    proc_start = MPI.Wtime()
    for job_name in my_jobs:
        analyze_res = image_analyze(job_name, ocr_config)
        my_result.append({
            'name': int(job_name.replace(PDF_NAME, '')),
            'text': analyze_res[0],
            'time': analyze_res[1]
        })
    print('[%d]Tasks completed!: %5.4fs.\nNow sending...' % (rank, MPI.Wtime() - proc_start))

    req = comm.isend(my_result, dest=0, tag=0)
    req.wait()
    # print('[%d]Sending completed!' % rank)
