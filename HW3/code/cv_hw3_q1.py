###############################################################################
#                            pkg imports                                      #
###############################################################################
import torch
import torchvision
import torchvision.transforms   as transforms
import torch.nn                 as nn
import os 
import matplotlib.pyplot        as plt
import matplotlib.cm as cm
import numpy                    as np
import cv2
from PIL                        import Image, ImageFilter



###############################################################################
#                               constants                                     #
###############################################################################
# filesystem related 
DIR_MY_DATA         = 'my_data'
DIR_DATA            = 'data'
SUBDIR_FROGS        = 'frogs'
SUBDIR_HORSES       = 'horses'
SUBDIR_RAND_IMAGES  = 'random_images'
IMG_COW             = 'cow.jpg'
IMG_SHEEP           = 'sheep.jpg'
IMG_BEACH           = 'beach.jpg'
TXT_LABELS          = 'imagenet1000_clsidx_to_labels.txt'
# model transform
VGG16_HEIGHT        = 224
VGG16_WIDTH         = 224
NORM_MEAN           = [0.485, 0.456, 0.406]
NORM_STD            = [0.229, 0.224, 0.225]
# model constants
DEEP_LAB_CLASSES    = 21
GRABCUT_CLASSES     = 2     # grabcut gives either background or foreground
# grabcut dictionary
GRABCUT_RECT_DICT   = {
    'frog1.jpg'     : (100,  71, 294, 270),
    'frog2.jpg'     : (148, 128, 192, 167),
    'horse1.png'    : ( 23,  10, 870, 554),
    'horse2.jpg'    : (577, 164, 438, 528),
    'q3_cat.jpeg'   : (  0,  24, 493, 491),
    'q3_kindle.jpg' : ( 76,  67, 362, 368),
    'q3_tv.jpg'     : ( 83,  63, 179, 123)
}
GRABCUT_ITER        = 5

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



###############################################################################
#                               transforms                                    #
###############################################################################
transform_deeplab = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean = NORM_MEAN, std = NORM_STD)
    ]
)

transform_vgg = transforms.Compose(
    [
        transforms.Resize((VGG16_HEIGHT, VGG16_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean = NORM_MEAN, std = NORM_STD)
    ]
)


###############################################################################
#                            functions                                        #
###############################################################################
def load_images(fp, named = False, method = 'PIL'):
    # desc
    # recieves a path to a dir of images and load them all to a list
    #
    # input
    # fp - path to dir
    #
    # output
    # im_list - list of PIL.Image objects
    img_list = []
    for fn_img in os.listdir(fp):
        fp_img = os.path.join(fp, fn_img)
        if method == 'cv':
            img_obj = cv2.imread(fp_img)
            img_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2RGB)
        elif method == 'PIL':
            img_obj = Image.open(fp_img)
        else:
            print('bad image method')
            return None
        if named:
            img_list.append((fn_img, img_obj))
        else:
            img_list.append(img_obj)
    return img_list


def wrapper_grabcut(named_img_list, colors_palette):
    seg = []
    masks = []

    for fn, img in named_img_list:
        mask        = np.zeros(img.shape[:2], np.uint8)
        bgd_model   = np.zeros((1, 65), np.float64)
        fgd_model   = np.zeros((1, 65), np.float64)
        rct         = GRABCUT_RECT_DICT[fn]
        cv2.grabCut(img, mask, rct, bgd_model, fgd_model, GRABCUT_ITER, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        mask_img = Image.fromarray(mask2)
        mask_img.putpalette(colors_palette)
        masked_img = img * mask2[:, :, np.newaxis]
        seg.append(masked_img)
        masks.append(mask_img)

    return seg, masks


def plot_classic_segments(seg_list, img_label):

    num_im = len(seg_list)
    _, axes = plt.subplots(1, num_im)
    if num_im == 1:
        img = seg_list[0]
        axes.imshow(img)
        axes.set_title(img_label + ': masked img')
        axes.set_xticks([])
        axes.set_yticks([])
    else:
        for i, img in enumerate(seg_list):
            axes[i].imshow(img)
            axes[i].set_title(img_label + ': seg. for img #' + str(i + 1))
            axes[i].set_xticks([])
            axes[i].set_yticks([])  
    plt.show()



def get_palette(classes):
    # desc
    # creates a color palette according to the number of classes of the 
    # model
    #
    # input
    # classes - number of classes the model can recognize
    #
    # output
    # colors - a palette of classes colors
    palette = torch.tensor([2 ** 25 -1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(classes)])[:, None] * palette
    colors = (colors % 255).numpy().astype('uint8')
    return colors



def predict_segment(model, img_list):
    # desc
    # takes in a model, list of images and a path to labels file and match
    # between the predicte class to the image
    #
    # input
    # model     - the model which evaluates the class
    # im_list   - list of images which class we predict
    # fp_labels - path to a labels file
    #
    # output
    # im_label list - list of tuples in the form of (image, predicted class)
    pred_list = []
    for img in img_list:
        img_preprocess = preprocess_image_deeplab(img)
        # img_preprocess = img_preprocess.to('cuda:0')
        pred_segment = predict_image_segment(model, img_preprocess)
        pred_list.append((img, pred_segment))
    return pred_list


def predict_image_segment(model, img):
    # desc
    # predicts the segments in a given image using model
    #
    # input
    # model     - model which predicts the image segments
    # img       - image which we want to segment
    #
    # output
    # 
    with torch.no_grad():
        output = model(img)['out'][0]
    return output.argmax(0)


def preprocess_image_deeplab(img):
    # desc
    # preprocess a single image to fit deeplab prerequisites                     
    #
    # input
    # img - a single image to preprocess
    #
    # output
    # a preprocessed image which can be forwarded to the model via model(img)
    return transform_deeplab(img).unsqueeze(0)


#def create_obj_mask(predicted_image):
#    BW_COLORS = np.array([[0,0,0], [255,255,255]])
#    img, segment = predicted_image[0]
#    r = Image.fromarray(segment.byte().cpu().numpy()).resize(img.size)
#    r.putpalette(BW_COLORS)
#    img = r.convert('1')
#    fig=plt.figure(figsize=(15,15))
#    ax=fig.add_subplot(111)
#    ax.imshow(img)
#    ax.set_axis_off()
#    plt.show()
#    return r



def plot_segments(pred_list, colors_palette, img_label):
    palettes = []
    for img, segment in pred_list:
        r = Image.fromarray(segment.byte().cpu().numpy()).resize(img.size)
        r.putpalette(colors_palette)
        palettes.append(r)

    num_im = len(palettes)
    _, axes = plt.subplots(1, num_im)
    for i, p in enumerate(palettes):
        axes[i].imshow(p)
        axes[i].set_title(img_label + ': seg. for img #' + str(i + 1))
        axes[i].set_xticks([])
        axes[i].set_yticks([])  
    plt.show()


def plot_object(pred_list, img_label):
    masked = []
    for img, segment in pred_list:
        mask = torch.zeros_like(segment).float().to(device)
        mask[segment != 0] = 1
        masked_img = img * mask.unsqueeze(2).byte().cpu().numpy()
        masked.append(masked_img)

    num_im = len(masked)
    _, axes = plt.subplots(1, num_im)
    if num_im == 1:
        img = masked[0]
        axes.imshow(img)
        axes.set_title(img_label + ': masked img')
        axes.set_xticks([])
        axes.set_yticks([])
    else:
        for i, p in enumerate(masked):
            axes[i].imshow(p)
            axes[i].set_title(img_label + ': masked img #' + str(i + 1))
            axes[i].set_xticks([])
            axes[i].set_yticks([])  
    plt.show()



def predict_class(model, im_list, fp_labels):
    # desc
    # takes in a model, list of images and a path to labels file and match
    # between the predicte class to the image
    #
    # input
    # model     - the model which evaluates the class
    # im_list   - list of images which class we predict
    # fp_labels - path to a labels file
    #
    # output
    # im_label list - list of tuples in the form of (image, predicted class)
    fh_labels = open(fp_labels, 'r')
    labels = eval(fh_labels.read())
    im_label_list = []
    for im in im_list:
        im_preprocess = preprocess_image_vgg16(im)
        label = predict_image_label(model, im_preprocess, labels)
        im_label_list.append((im, label))
    return im_label_list


def preprocess_image_vgg16(im):
    # desc
    # preprocess a single image to fit vgg16 prerequisites                     
    #
    # input
    # im - a single image to preprocess
    #
    # output
    # a preprocessed image which can be forwarded to the model via model(im)
    return transform_vgg(im).unsqueeze(0)


def predict_image_label(model, im, labels):
    # desc
    # predicts the image class and assigns a string label
    #
    # input
    # model     - model which predicts the label num
    # im        - image which class we predict
    # labels    - an array of labels per model class
    #
    # output
    # im_label - a string describing the predicted classs
    _, i = model(im).data[0].max(0)
    i = i.numpy().item()
    im_label = labels[i]
    return im_label
       

def plot_prediction(pred_list):
    # desc
    # takes a list of tuples in the form of (image, label) and plots the images
    # with a proper title corresponding to the label
    #
    # input
    # pred_list - list of predicted labels in the form of (image, label)
    num_im = len(pred_list)
    _, axes = plt.subplots(1, num_im)
    if num_im == 1:
        (im, label) = pred_list[0]
        axes.imshow(im)
        axes.set_title(label)
        axes.set_xticks([])
        axes.set_yticks([])
    else:
        for i, (im, label) in enumerate(pred_list):
            axes[i].imshow(im)
            axes[i].set_title(label)
            axes[i].set_xticks([])
            axes[i].set_yticks([])  
    plt.show()




def paste_image(front, bg, os, mask, resize):
    if resize is not False:
        front.thumbnail(resize)
        mask.thumbnail(resize)
    bg.paste(front, os, mask)
    return bg


def binary_mask(img, mask, colors_palette):
    img_mask = Image.fromarray(mask.byte().cpu().numpy()).resize(img.size)
    img_mask.putpalette(colors_palette)
    bw_mask = img_mask.convert('1')
    return bw_mask

###############################################################################
#                                main                                         #
###############################################################################
def main():
    root = os.path.dirname(__file__)
    fp_labels = os.path.join(root, DIR_DATA, TXT_LABELS)

    # SUBQUESTION 1.1
    fp_frogs = os.path.join(root, DIR_DATA, SUBDIR_FROGS)
    fp_horses = os.path.join(root, DIR_DATA, SUBDIR_HORSES)
    img_frogs = load_images(fp_frogs)
    img_horses = load_images(fp_horses)
    img_named_frogs = load_images(fp_frogs, named = True, method = 'cv')
    img_named_horses = load_images(fp_horses, named = True, method = 'cv')

    deeplab_palette = get_palette(DEEP_LAB_CLASSES)
    grabcut_palette = get_palette(GRABCUT_CLASSES)


    # SUBQUESTION 1.2 - CLASSIC METHOD
    #classic_frogs_seg, classic_frogs_masks = wrapper_grabcut(img_named_frogs, grabcut_palette)
    #classic_horse_seg, classic_horse_masks = wrapper_grabcut(img_named_horses, grabcut_palette)
    #plot_classic_segments(classic_frogs_seg, 'frogs')
    #plot_classic_segments(classic_frogs_masks, 'frogs')
    #plot_classic_segments(classic_horse_masks, 'horses')
    #plot_classic_segments(classic_horse_seg, 'horses')

    
    # SUBQUESTION 1.2 - DEEP LEARNING BASED METHOD
    model_deeplab = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained = True)
    model_deeplab.eval()
    model_deeplab = model_deeplab.to(device)
    #pred_frogs = predict_segment(model_deeplab, img_frogs)
    #plot_segments(pred_frogs, deeplab_palette, 'frogs')
    #plot_object(pred_frogs, 'frogs')
    #pred_horses = predict_segment(model_deeplab, img_horses)
    #plot_segments(pred_horses, deeplab_palette, 'horses')
    #plot_object(pred_horses, 'horses')

    # SUBQUESTION 1.3 and 1.4
    #fp_rnd_images = os.path.join(root, DIR_MY_DATA, SUBDIR_RAND_IMAGES)
    #img_rnd_images = load_images(fp_rnd_images)
    #img_named_rnd_images = load_images(fp_rnd_images, named = True, method = 'cv')
    #classic_rnd_seg, classic_rnd_masks = wrapper_grabcut(img_named_rnd_images, grabcut_palette)
    #plot_classic_segments(classic_rnd_seg, 'SQ3')
    #plot_classic_segments(classic_rnd_masks, 'SQ3')
    
    #pred_rnd = predict_segment(model_deeplab, img_rnd_images)
    #plot_segments(pred_rnd, deeplab_palette, 'SQ3')
    #plot_object(pred_rnd, 'SQ3')


    # SUBQUESTION 1.5
    # TODO: find pre or post processing to improve edges

    # SUBQUESTION 1.6
    model_vgg16 = torchvision.models.vgg16(pretrained = True, progress = True)
    model_vgg16.eval()
    model_vgg16 = model_vgg16.to(device)


    # SUBQUESTION 1.7
    fp_cow = os.path.join(root, DIR_DATA, IMG_COW)
    img_cow = Image.open(fp_cow) 
    #fp_sheep = os.path.join(root, DIR_DATA, IMG_SHEEP)
    #img_sheep = Image.open(fp_sheep)
    #img_list = [img_cow, img_sheep]
    #pred_animals = predict_class(model_vgg16, img_list, fp_labels)
    #plot_prediction(pred_animals)

    # SUBQUESTION 1.8
    #pred_cow = predict_segment(model_deeplab, [img_cow])
    #plot_object(pred_cow, 'cow')

    # SUBQUESTION 1.9
    fp_beach = os.path.join(root, DIR_DATA, IMG_BEACH)
    img_beach = Image.open(fp_beach) 
    pred_cow = predict_segment(model_deeplab, [img_cow])
    _, cow_mask = pred_cow[0]
    binary_mask_cow = binary_mask(img_cow, cow_mask, deeplab_palette)
    cow_on_the_beach = paste_image(
        front = img_cow, 
        bg = img_beach, 
        os = (300, 150),
        mask = binary_mask_cow,
        resize = (250, 250)
    )
    _, axes = plt.subplots(1, 1)
    axes.imshow(cow_on_the_beach)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.show()

    # SUBQUESTION 1.10
    pred_beach_cow = predict_class(model_vgg16, [cow_on_the_beach], fp_labels)
    plot_prediction(pred_beach_cow)




if __name__ == "__main__":
    main()