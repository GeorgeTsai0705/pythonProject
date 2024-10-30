import fitz
import os
import time
from os import path

wafer_ID = "2224-111"

# 打開PPT的空白模板
PDF_base = fitz.open("Original.pdf")
# List the folder and filename
dirs = [d for d in os.listdir() if path.isdir(d)]

# Setting the Rect Working Area
rects = []
Rect_para = {'init_x': 10, 'init_y': 100, 'x_band': 225, 'y_band': 230, 'col_band': 10, 'row_band': 200}
for j in range(2):
    for i in range(4):
        X = Rect_para['x_band'] + Rect_para['col_band']
        Y = Rect_para['row_band']
        rects.append(fitz.Rect(Rect_para['init_x'] + i * X,
                               Rect_para['init_y'] + j * Y,
                               Rect_para['init_x'] + Rect_para['x_band'] + i * X,
                               Rect_para['init_y'] + Rect_para['y_band'] + j * Y))

# Chip order setting
chip_order = {'D1': 1, 'D2': 2, 'C1': 3, 'C2': 4, 'C3': 5, 'C4': 6, 'C5': 7, 'C6': 8, 'C7': 9, 'C8': 10, 'B1': 11,
              'B2': 12, 'B3': 13, 'B4': 14, 'B5': 15, 'B6': 16, 'B7': 17, 'A2': 18, 'A3': 19}


def insert_wafer_map(page, dir_name, chip_state):
    area = page.bound()
    # if chip already finished Screen process, add an image of Al reflectivity result at page 0
    if chip_state == "Screen":

        # insert wafermap image
        area[2] = area[2] / 2
        insert_image = open(path.join(dir_name, f"{wafer_ID}.png"), "rb").read()
        page.insert_image(area, stream=insert_image)

        # insert an image of Al reflectivity result
        area[0] = area[2]
        area[2] = area[2] * 2
        insert_image = open(path.join("Al_Reflectivity2.png"), "rb").read()
        page.insert_image(area, stream=insert_image)

    else:
        # insert wafermap image
        area[0] = (area[0] + area[2]) / 8
        insert_image = open(path.join(dir_name, f"{wafer_ID}.png"), "rb").read()
        page.insert_image(area, stream=insert_image)

    return page


def insert_chip_image(doc, dir_name, chip_dir, status):
    # prepare new page in the end of document
    doc.fullcopy_page(pno=0, to=-1)
    # list image file name under the chip_dir folder
    image_list = os.listdir(path.join(dir_name, status, chip_dir))

    photo_num = 0
    if status == "Photolithography_Particles":
        for ele in image_list:
            photo_num = int(ele.replace(".jpg", ""))
            insert_image = open(path.join(dir_name, status, chip_dir, ele), "rb").read()
            doc[-1].insert_image(rects[photo_num], stream=insert_image)

    elif status == "SEM":
        for ele in image_list:
            photo_num = int(ele.replace(".BMP", ""))
            insert_image = open(path.join(dir_name, status, chip_dir, ele), "rb").read()
            doc[-1].insert_image(rects[photo_num], stream=insert_image)

    elif status == "Screen":
        area = doc[-1].bound()
        for ele in image_list:
            if ele.find('_Image') >= 0:
                insert_image = open(path.join(dir_name, status, chip_dir, ele), "rb").read()
                doc[-1].insert_image((area[2] / 2, 0, area[2] , area[3]), stream=insert_image)

            elif ele.find('_FullImage') >= 0:
                insert_image = open(path.join(dir_name, status, chip_dir, ele), "rb").read()
                doc[-1].insert_image((0, 0, area[2] / 2, area[3]), stream=insert_image)


    else:
        # if the filename of slit's image is A1.jpg
        if image_list[-1][0].isalpha():
            # deal the slit's image independently
            slit_image = open(path.join(dir_name, status, chip_dir, image_list.pop()), "rb").read()
            doc[-1].insert_image(rects[0], stream=slit_image)

        # insert the others images
        for ele in image_list:
            insert_image = open(path.join(dir_name, status, chip_dir, ele), "rb").read()
            photo_num = int(ele.replace(".jpg", "")) - 1
            doc[-1].insert_image(rects[photo_num], stream=insert_image)
    return doc


def insert_chip_name(doc, chip_dir):
    # insert the chip name
    T = fitz.TextWriter(page_rect=(10, 10, 100, 100))
    font = fitz.Font(fontname="tibo", is_bold=True)
    if len(chip_dir) <20:
        chip_name = chip_dir[0:11]
    else:
        chip_name = chip_dir[11:22] + "(NEW)"
    T.append((10, -380), chip_name, font, fontsize=36)
    T.write_text(doc[-1])
    return doc


def check_folder(dir_name, folder_name):
    l = []
    if os.path.isdir(path.join(dir_name, folder_name)):
        l = os.listdir(path.join(dir_name, folder_name))
    return l


def insert_chip_info(doc, dir_name, chip_dir, txt_name):
    # Store Operation Date
    OP_date = dir_name[0:10]

    # load txt file
    Chip_Name = []
    Slit = []
    Type = []
    Status = []

    with open(path.join(dir_name, txt_name), "r", encoding='utf-8') as f:
        for line in f.readlines():
            s = line.split('\t')
            Chip_Name.append(s[0])
            Type.append(s[2])
            Status.append(s[3])
            Slit.append(int(s[4]))

    chip_cont = Chip_Name.index(chip_dir[20:22])

    buffer_ = f" Date: {OP_date}\n" \
              f" Slit: {Slit[chip_cont]} um\n" \
              f" Status: {Type[chip_cont]} {Status[chip_cont]}"
    rect_ = (420, 20, 820, 120)
    shape = doc[-1].new_shape()
    shape.insert_textbox(rect=rect_, buffer=buffer_, fontsize=24, fontname="tibo")
    shape.draw_rect(rect=rect_)
    shape.finish(color=(0, 0, 0), width=2)
    shape.commit()
    return doc



def insert_chip_structure_image(dir_name):
    # check folder: before coating
    chip_dirs_list = check_folder(dir_name, folder_name= "Photolithography")

    # check folder: before coating particles
    chip_p_dirs_list = check_folder(dir_name, folder_name= "Photolithography_Particles")

    # check folder: SEM
    chip_s_dirs_list = check_folder(dir_name, folder_name= "SEM")

    # check folder: after coating
    chip_ac_dirs_list = check_folder(dir_name, folder_name= "coating")

    # check folder: Dicing
    chip_d_dirs_list = check_folder(dir_name, folder_name= "Dicing")

    # check folder: Screen
    chip_scr_dirs_list = check_folder(dir_name, folder_name= "Screen")

    # sort the chip order as we set before
    chip_dirs_list.sort(key=lambda s: chip_order[s[20:22]])

    # initial chip state as Before_Coating, use this parameter to check chip state
    chip_state = "Photolithography"

    for chip_dir in chip_dirs_list:
        # insert images to document
        insert_chip_image(PDF_base, dir_name, chip_dir, "Photolithography")
        insert_chip_name(PDF_base, chip_dir)
        insert_chip_info(PDF_base, dir_name, chip_dir, f"{wafer_ID}.txt")

        # Check this chip for particle images?
        if chip_dir in chip_p_dirs_list:
            insert_chip_image(PDF_base, dir_name, chip_dir, "Photolithography_Particles")
            insert_chip_name(PDF_base, chip_dir)
            insert_chip_info(PDF_base, dir_name, chip_dir, f"{wafer_ID}.txt")

        # Check this chip for an SEM image?
        if chip_dir in chip_s_dirs_list:
            insert_chip_image(PDF_base, dir_name, chip_dir, "SEM")
            insert_chip_name(PDF_base, chip_dir)

        # if this chip has already coated, then insert the coated image
        for ac_dir_name in chip_ac_dirs_list:
            if chip_dir[11:22] == ac_dir_name[11:22]:
                insert_chip_image(PDF_base, dir_name, ac_dir_name, "Coating")
                insert_chip_name(PDF_base, ac_dir_name)
                insert_chip_info(PDF_base, dir_name, ac_dir_name, f"{wafer_ID}-AC.txt")
                chip_state = "Coating"
                break

        # if this chip has already diced, then insert the diced image
        for d_dir_name in chip_d_dirs_list:
            if chip_dir[11:22] == d_dir_name[11:22]:
                insert_chip_image(PDF_base, dir_name, d_dir_name, "Dicing")
                insert_chip_name(PDF_base, d_dir_name)
                insert_chip_info(PDF_base, dir_name, d_dir_name, f"{wafer_ID}-D.txt")
                chip_state = "Dicing"
                break

        for src_dir_name in chip_scr_dirs_list:
            if chip_dir[11:22] == src_dir_name[0:11]:
                insert_chip_image(PDF_base, dir_name, src_dir_name, "Screen")
                insert_chip_name(PDF_base, src_dir_name)
                chip_state = "Screen"
                break

    return chip_state


def main():
    for dir_name in dirs:
        if wafer_ID in dir_name:
            chip_state = insert_chip_structure_image(dir_name)

            # insert the wafer map at the beginning of PDF
            insert_wafer_map(PDF_base[0], dir_name, chip_state)
            break

    localtime = time.localtime()
    TimeText = time.strftime("%m%d", localtime)

    PDF_base.save(f"output/{wafer_ID}-{TimeText}-{chip_state}.pdf")


if __name__ == "__main__":
    main()
