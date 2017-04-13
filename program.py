import sys
import command_line_utils as clu
import image_utils as iu
import cv2
import contour_classifier as cc

def main(args):
    if not clu.check_command_line_arguments(args):
        exit()
    directory_path = args[1]
    pictures_no = int(args[2])
    pictures = iu.load_pictures(directory_path, pictures_no)
    normalized_figures = iu.normalize_figures(pictures, 300)
    #for i in range(pictures_no):
    #    iu.show_image(str(i), normalized_figures[i])
    classification = cc.get_classification(normalized_figures)
    printResults(classification)
    #print classification


    #cont = iu.get_contour(normalized_figures[0])
    #cont = cv2.approxPolyDP(cont, 2,True)
    #cont = [value for value in cont if value>30]
    #iu.show_image("im", normalized_figures[0])
    #imtemp = normalized_figures[0]
    #iu.show_image("sldkfj", imtemp)
    #img2 = cv2.drawContours(imtemp, cont, -1,(255,0,0))
    #drawContours(img, contours, -1, RGB(255, 0, 0), 1.5, 8, hierarchy, 2, Point());
    #iu.show_image("imfsdf", img2)
    #print(cont)

def printResults(classification):
    for i in range(len(classification)):
        print(' '.join([str(x) for x in classification[i]]))


def fixed_params():
    return ['.', 'set0', '6']


def sys_params():
    return sys.argv

if __name__ == '__main__':
    main(sys_params())
