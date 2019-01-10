""" Tesseract OCR Test Program within MPI """
# coding: UTF-8

import sys
import os
from operator import itemgetter
from PIL import Image
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


def ocr(jn, rk):
    # 利用できるOCRエンジン
    tools = pyocr.get_available_tools()
    if not tools:
        print('No OCR tool found.')
        sys.exit(1)
    # Tesseract OCRを選択
    my_tool = tools[0]
    print("Will use tool '%s'" % (my_tool.get_name()))
    # 利用できる言語
    langs = my_tool.get_available_languages()
    print("Available languages: %s" % ", ".join(langs))
    # 日本語を選択
    my_lang = 'jpn'
    print("Will use lang {}".format(my_lang))

    path = IMG_DIR + str(jn) + IMG_TYPE
    print('Now analyzing {} from No.{}'.format(path, rk))
    raw_text = my_tool.image_to_string(
        Image.open(path),
        lang=my_lang,
        builder=pyocr.builders.TextBuilder()
    )
    print("Analyzing Finished from {}: {}".format(rk, raw_text))
    raw_text2 = regex.sub(r'((\s)*((\p{Han})|(\p{Hiragana})|(\p{Katakana})|[、。？！])|(\n){2})', r'\3', raw_text)
    text = regex.sub(r'\n|([ ]+)', r' ', raw_text2)
    return text


i = 2
# number of pages
PAGE_NUM = i + 1
res_data = []
print('Number of Pages: ' + str(PAGE_NUM))

# get size of files and store them
for i in range(0, PAGE_NUM):
    file_size = os.path.getsize('{}{}{}{}'.format(IMG_DIR, PDF_NAME, i, IMG_TYPE))
    res_data.append({'name': PDF_NAME + str(i), 'size': round(file_size, 2)})
# sort by filesize ascending
file_data_s = sorted(res_data, key=itemgetter('size'), reverse=True)

my_jobs = []
my_num = len(jobs[0])
for i in range(0, my_num):
    my_jobs.append(jobs[0][i]['name'])
print('Received data from {}'.format(rank))
print(my_jobs)

master_result = []
for job_name in my_jobs:
    master_result.append({'name': job_name, 'text': ocr(job_name, rank)})

print('Task complete from 0.')

# Receive from workers
for i in range(1, size):
    print("Now receiving from {}".format(i))
    req = comm.irecv(source=i, tag=i)
    recv_result = req.wait()
    print("Received from {}".format(i))
    master_result.append(recv_result)

print('All received.')
data_s = sorted(master_result, key=itemgetter('name'))
print(data_s)
