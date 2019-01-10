import sys
from pdf2image import convert_from_path

fn = str(sys.argv[1])
images = convert_from_path('pdf/{}.pdf'.format(fn))

i = 1
for image in images:
  image.save('pdf2img/{}{}.png' . format(fn, i), 'png')
  i += 1