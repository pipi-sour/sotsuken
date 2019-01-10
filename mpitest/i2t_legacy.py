""" Tesseract OCR Test Program without MPI """
# coding: UTF-8

import sys
import io
import time
from PIL import Image
import regex
import pyocr
import pyocr.builders

#sys.stdin  = io.TextIOWrapper(sys.stdin.buffer,  encoding='utf-8')
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
#sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

fn = str(sys.argv[1])

start_t = time.time()
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

text = ''
for i in range (1, 3):
    text_tmp = tool.image_to_string(
        Image.open('img/{}{}.png'.format(fn, i)),
        lang=lang,
        builder=pyocr.builders.TextBuilder()
    )
    text_tmp = regex.sub(r'((\s)*((\p{Han})|(\p{Hiragana})|(\p{Katakana})|[、。？！])|(\n){2})', r'\3', text_tmp)
    text = regex.sub(r'\n|([ ]+)', r' ', text_tmp)
    if(text == ''):
        text = text_tmp
    else:
        text += '\n' + text_tmp

print(u"--- 結果 ---\n" + text)

f = open('txt/' + fn + '.txt', 'w')
text.encode('utf-8')
f.write(text)

print("全実行時間: %5.4fs." % (time.time() - start_t))