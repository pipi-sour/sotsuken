""" Tesseract OCR Test Program within MPI """
# coding: UTF-8

import sys
import io
from PIL import Image
from mpi4py import MPI
from pdf2image import convert_from_path
import regex
import pyocr
import pyocr.builders

#sys.stdin  = io.TextIOWrapper(sys.stdin.buffer,  encoding='utf-8')
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
#sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

fn = str(sys.argv[1])

MasterNode = 0

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()
procName = MPI.Get_processor_name()

print("Running from processor %s, rank %d out of %d processors." % (procName, rank, size))

comm.Barrier()

if rank == MasterNode:
    whole_start = MPI.Wtime()
    """
    images = convert_from_path('pdf/{}.pdf'.format(fn))
    i = 0
    for image in images:
        image.save('pdf2img/{0}{1}.png' . format(fn, i+1), 'png')
        i += 1
    #画像の枚数をnimgに格納
    nimg = i
    """
    nimg = 2
    req = []
    for j in range(1, size):
        if j > nimg:
            print('Now sending empty message to {}'.format(j))
            req.append(comm.isend('0', dest=j, tag=j))

        else:
            print('Now sending {0}{1}.png to {2}'.format(fn, j, j))
            req.append(comm.isend('pdf2img/{0}{1}.png'.format(fn, j), dest=j, tag=j))
        req[j-1].wait()

comm.Barrier()

if rank != MasterNode:
    proc_start = MPI.Wtime()
    rreq = comm.irecv(source=0, tag=rank)
    data = rreq.wait()
    if data == '0':
        print('Now sending 0 to master from {}'.format(rank))
        """
        sreq = comm.isend('0', dest=0, tag=rank)
        sreq.wait()
        """
        comm.send('0', dest=0, tag=rank)
    else:
        print('Now Analyzing {} from {}'.format(data, rank))
        
        # 利用できるOCRエンジン
        tools = pyocr.get_available_tools()
        if not tools:
            print("No OCR tool found")
            sys.exit(1)

        # Tesseract OCRを選択
        tool = tools[0]
        print("Will use tool '%s'" % (tool.get_name()))

        # 利用できる言語
        langs = tool.get_available_languages()
        print("Available languages: %s" % ", ".join(langs))

        # 日本語を選択
        lang = "jpn"
        print("Will use lang '%s'" % (lang))

        txt = tool.image_to_string(
            Image.open(data),
            lang=lang,
            builder=pyocr.builders.TextBuilder()
        )
        print('Now sending to master from {}'.format(rank))
        """
        sreq = comm.isend(txt, dest=0, tag=rank)
        sreq.wait()
        """
        comm.send(txt, dest=0, tag=rank)

    print('Sending Complete from {}'.format(rank))
    print("プロセス%d実行時間: %5.4fs" % (rank, MPI.Wtime() - proc_start))
    MPI.Finalize()

if rank == MasterNode:
    text = ''
    for i in range(1, size):
        print("Now Received from {}".format(i))
        req = comm.irecv(source=i, tag=i)
        data = req.wait()
        if data != '0':
            data = regex.sub(r'((\s)*((\p{Han})|(\p{Hiragana})|(\p{Katakana})|[、。？！])|(\n){2})', r'\3', data)
            data = regex.sub(r'\n|([ ]+)', r' ', data)
            if text == '':
                text = data
            else:
                text += '\n' + data
            print(text + ' is appended.')

    print(u"--- 結果 ---\n" + text)

    f = open('txt/' + fn + '.txt', 'w')
    text.encode('utf-8')
    f.write(text)
    print(u"全実行時間: %5.4fs." % (MPI.Wtime() - whole_start))
