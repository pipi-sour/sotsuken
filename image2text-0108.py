""" Tesseract OCR Test Program within MPI """
# 確実に動くプログラム☆
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

# sys.stdin  = io.TextIOWrapper(sys.stdin.buffer,  encoding='utf-8')
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

PDF_NAME = str(sys.argv[1])
PDF_DIR = 'pdf/'
IMG_DIR = 'pdf2img/'
IMG_TYPE = '.png'
# MasterNode's Proc No.
MASTER_NODE = 0

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
proc_name = MPI.Get_processor_name()
print("Running from processor %s, rank %d out of %d processors." % (proc_name, rank, size))
comm.Barrier()


def ocr(jn, rk):
    # 利用できるOCRエンジン
    tools = pyocr.get_available_tools()
    if not tools:
        print('[{}{}]No OCR tool found.'.format(rk, proc_name))
        sys.exit(1)
    # Tesseract OCRを選択
    my_tool = tools[0]
    # print('[{}]Will use tool {}'.format(rk, my_tool.get_name()))
    # 利用できる言語
    # langs = my_tool.get_available_languages()
    # print("Available languages: %s" % ", ".join(langs))
    # 日本語を選択
    my_lang = 'jpn'
    # print("[{}]Will use lang {}".format(rk, my_lang))

    path = IMG_DIR + str(jn) + IMG_TYPE
    print('[{}]Analyzing {}...'.format(rk, path))
    raw_text = my_tool.image_to_string(
        Image.open(path),
        lang=my_lang,
        builder=pyocr.builders.TextBuilder()
    )
    raw_text2 = regex.sub(r'((\s)*((\p{Han})|(\p{Hiragana})|(\p{Katakana})|[、。？！])|(\n){2})', r'\3', raw_text)
    text = regex.sub(r'\n|([ ]+)', r' ', raw_text2)
    # print("[{}]Analyzing finished!: {}".format(rk, text))
    print("[{}]Analyzing finished!".format(rk))
    return text


if rank == MASTER_NODE:
    whole_start = MPI.Wtime()

    pdf_images = convert_from_path('{0}{1}.pdf'.format(PDF_DIR, PDF_NAME))
    i = 0
    for image in pdf_images:
        image.save('{0}{1}{2}{3}'.format(IMG_DIR, PDF_NAME, i, IMG_TYPE))
        i += 1

    # number of pages
    PAGE_NUM = i
    res_data = []
    print('Number of Pages: ' + str(PAGE_NUM))

    # get size of files and store them
    for i in range(PAGE_NUM):
        file_size = os.path.getsize('{}{}{}{}'.format(IMG_DIR, PDF_NAME, i, IMG_TYPE))
        res_data.append({'name': PDF_NAME + str(i), 'size': round(file_size, 2)})
    # sort by filesize ascending
    file_data_s = sorted(res_data, key=itemgetter('size'), reverse=True)

    # jobs each workers
    jobs = []
    for i in range(0, size):
        jobs.append([])
    # sizes of images
    worker_size = [0] * size

    # job equalizing
    for i in range(0, PAGE_NUM):
        min_size_index = worker_size.index(min(worker_size))
        jobs[min_size_index].append(file_data_s[i])
        worker_size[min_size_index] += file_data_s[i].get("size")
    for i in range(0, size):
        print("[{}]{}\tlength: {}({})".format(i, jobs[i], len(jobs[i]), worker_size))

    req = []
    for i in range(size):
        req.append([])
    # Send workers
    for i in range(1, size):
        # print('Now sending to ' + str(i))
        comm.send(len(jobs[i]), dest=i, tag=i)
        print('jobs[{}]: {}'.format(i, jobs[i]))
        for j in range(0, len(jobs[i])):
            msg = jobs[i][j]['name']
            req[i].append(comm.isend(msg, dest=i, tag=i))
            # print('MSG to {}: {}, {}'.format(i, msg, req[i][j]))
            req[i][j].wait()
            print('SENT!')

    proc_start = MPI.Wtime()

    my_jobs = []
    my_num = len(jobs[0])
    for i in range(0, my_num):
        my_jobs.append(jobs[0][i]['name'])
    print('Jobs of master: {}'.format(my_jobs))

    master_result = []
    for job_name in my_jobs:
        master_result.append({'name': int(job_name.replace(PDF_NAME, '')), 'text': ocr(job_name, rank)})

    print('Task complete from 0.')
    print("プロセス%d実行時間: %5.4fs" % (rank, MPI.Wtime() - proc_start))

if rank != MASTER_NODE:
    proc_start = MPI.Wtime()

    my_jobs = []
    req = comm.irecv(source=0, tag=rank)
    my_num = req.wait()
    req = []
    print('[{}]N of job: {}\nReceiving...'.format(rank, my_num))
    tmp_res = []
    for i in range(0, my_num):
        req.append(comm.irecv(source=0, tag=rank))
        my_jobs.append(req[i].wait())
        print(my_jobs)
    print('[{}]Received!'.format(rank))
    # print(my_jobs)

    my_result = []
    for job_name in my_jobs:
        my_result.append({'name': int(job_name.replace(PDF_NAME, '')), 'text': ocr(job_name, rank)})

    print('Task complete from {}. Now sending...'.format(rank))
    req = comm.isend(my_result, dest=0, tag=0)
    req.wait()
    print('Sending complete from {}'.format(rank))
    print("プロセス%d実行時間: %5.4fs" % (rank, MPI.Wtime() - proc_start))

if rank == MASTER_NODE:
    # Receive from workers
    for i in range(1, size):
        print("Now receiving from {}".format(i))
        req = comm.irecv(source=i, tag=0)
        recv_result = req.wait()
        print("[0]Received!!".format(i))
        print('[{}]JOB_NUM: {}'.format(i, len(recv_result)))
        for j in range(len(recv_result)):
            master_result.append(recv_result[j])

    print('All received!')
    res_s = sorted(master_result, key=itemgetter('name'))
    # print(res_s)

    TXT_DIR = 'txt/' + PDF_NAME + '.txt'
    f = open(TXT_DIR, 'w')

    for i in range(PAGE_NUM):
        res_s[i]['text'].encode('utf-8')
        f.write('{}{}{}:\n\n{}\n\n\n'.format(PDF_NAME, i, IMG_TYPE, res_s[i]['text']))
    f.close()
    print('Writing Complete!\nLocation:{}'.format(TXT_DIR))
    print(u"Conglatulations!: %5.4fs." % (MPI.Wtime() - whole_start))

