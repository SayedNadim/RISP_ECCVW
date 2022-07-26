import ntpath
import sys
import os


# helper functions

# file helpers
import cv2


def path_leaf(path, image=True, cut_off_value=-4):
    head, tail = ntpath.split(path)
    if image:
        return tail[:cut_off_value] or ntpath.basename(head)
    else:
        return tail or ntpath.basename(head)


def default_flist_reader(flist):
    """
    flist format: dirath\ndirpath\n ...
    """
    im_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            imgs = line.strip().split()
            im_list.append(imgs)

    return im_list


def find_samples_in_subfolders(dir):
    """
    Finds the class folders in a dataset.
    Args:
        dir (string): Root directory path.
    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
    Ensures:
        No class is a subdirectory of another.
    """
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    samples = []
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if allowed_image_extensions(fname):
                    path = os.path.join(root, fname)
                    # item = (path, class_to_idx[target])
                    # samples.append(item)
                    samples.append(path)
    return samples


# array image functions
def allowed_image_extensions(filename):
    """
    Returns image files if they have the allowed image extensions
    :param filename: image file
    :return: image file
    """
    img_ext = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tiff', '.tif']
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in img_ext)


if __name__ == '__main__':
    mono_save_dir = '/home/la-belva/dataset/Zurich/Zurich-RAW-to-DSLR-Dataset/train/huawei_mono'
    visualized_image_dir = '/home/la-belva/dataset/Zurich/Zurich-RAW-to-DSLR-Dataset/train/huawei_visualized'
    image_list = [os.path.join(visualized_image_dir, x) for x in os.listdir(visualized_image_dir) if
                  allowed_image_extensions(x)]

    print(image_list)
    # print(path_leaf(image_list[0]))
    for index in range(len(image_list)):
        image = cv2.imread(image_list[index], cv2.IMREAD_GRAYSCALE)
        image_sl = path_leaf(image_list[index])
        cv2.imwrite(mono_save_dir + '/{}.png'.format(image_sl), image)
