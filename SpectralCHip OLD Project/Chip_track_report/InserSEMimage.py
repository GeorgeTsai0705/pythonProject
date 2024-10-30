import fitz
import os
from os import path

filename = "2224-101v2v2v2v2.pdf"
insertpage_num = 3
image_list = os.listdir("Image")
chipinfo = "2224-101-D2(NEW)"

PDF_base = fitz.open(filename)
Blank_base = fitz.open("Original.pdf")


def write(page, chipinfo):
    T = fitz.TextWriter(page_rect=(10, 10, 100, 100))
    font = fitz.Font(fontname="tibo", is_bold=True)
    T.append((10, -380), chipinfo, font, fontsize=36)
    T.write_text(page)

area = Blank_base[0].bound()
for i in range((len(image_list)-1) // 8 + 1):
    PDF_base.new_page(pno=insertpage_num, width=area[2], height=area[3])
    write(PDF_base[insertpage_num], chipinfo)

# Setting the Rect Working Area
Rects = []
Rect_para = {'init_x': 10, 'init_y': 100, 'x_band': 225, 'y_band': 230, 'col_band': 10, 'row_band': 200}
for j in range(2):
    for i in range(4):
        X = Rect_para['x_band'] + Rect_para['col_band']
        Y = Rect_para['row_band']
        Rects.append(fitz.Rect(Rect_para['init_x'] + i * X,
                               Rect_para['init_y'] + j * Y,
                               Rect_para['init_x'] + Rect_para['x_band'] + i * X,
                               Rect_para['init_y'] + Rect_para['y_band'] + j * Y))

photonum = 0

for image_name in image_list:
    page = PDF_base[insertpage_num + photonum // 8]
    if photonum < 8:
        insert_image = open(path.join("Image", image_name), "rb").read()
        page.insert_image(Rects[photonum % 8], stream=insert_image)

    else:
        insert_image = open(path.join("Image", image_name), "rb").read()
        page.insert_image(Rects[photonum % 8], stream=insert_image)
    photonum += 1

PDF_base.save(filename.replace(".pdf", "v2.pdf"))
