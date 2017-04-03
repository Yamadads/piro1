import sys
import command_line_utils as clu
import image_utils as iu


def main(args):
    if not clu.check_command_line_arguments(args):
        exit()
    directory_path = args[1]
    pictures_no = int(args[2])
    pictures = iu.load_pictures(directory_path, pictures_no)

    contour = iu.get_contour(pictures[0])

    for i in xrange(pictures_no):
        iu.get_normalized_figure(pictures[i], 300)

    #iu.get_normalized_figure(pictures[2], 300)
    #iu.show_image(pictures[0])


if __name__ == '__main__':
    main(sys.argv)
