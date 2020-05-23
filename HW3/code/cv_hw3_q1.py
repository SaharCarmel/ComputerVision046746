###############################################################################
#                            pkg imports                                      #
###############################################################################
import torch
import torchvision
import torchvision.transforms   as transforms
import torch.nn                 as nn
import os 
import matplotlib.pyplot        as plt
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
def load_images(fp):
    # desc
    # recieves a path to a dir of images and load them all to a list
    #
    # input
    # fp - path to dir
    #
    # output
    # im_list - listen of PIL.Image objects
    img_list = []
    for img in os.listdir(fp):
        fp_img = os.path.join(fp, img)
        img_obj = Image.open(fp_img)
        img_list.append(img_obj)
    return img_list


def blur_images(img_list):
    blurred = []
    for img in img_list:
        blr_img = img.filter(ImageFilter.BLUR)
        blurred.append(blr_img)
    return blurred



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

    # SUBQUESTION 1.2 - CLASSIC METHOD
    # TODO: not implemented yet
    
    
    # SUBQUESTION 1.2 - DEEP LEARNING BASED METHOD
    model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained = True)
    model.eval()
    model = model.to(device)
    deeplab_palette = get_palette(DEEP_LAB_CLASSES)
    #pred_frogs = predict_segment(model, img_frogs)
    #plot_segments(pred_frogs, deeplab_palette, 'frogs')
    #plot_object(pred_frogs, 'frogs')
    #pred_horses = predict_segment(model, img_horses)
    #plot_segments(pred_horses, deeplab_palette, 'horses')
    #plot_object(pred_horses, 'horses')

    # SUBQUESTION 1.3
    # fp_rnd_images = os.path.join(root, DIR_MY_DATA, SUBDIR_RAND_IMAGES)
    # img_rnd_images = load_images(fp_rnd_images)
    # pred_rnd = predict_segment(model, img_rnd_images)
    # plot_segments(pred_rnd, deeplab_palette, 'SQ3')
    # plot_object(pred_rnd, 'SQ3')


    # SUBQUESTION 1.5
    # TODO: find pre or post processing to improve edges

    # SUBQUESTION 1.6
    #model = torchvision.models.vgg16(pretrained = True, progress = True)
    #model.eval()
    #model = model.to(device)


    # SUBQUESTION 1.7
    fp_cow = os.path.join(root, DIR_DATA, IMG_COW)
    img_cow = Image.open(fp_cow)    
    #fp_sheep = os.path.join(root, DIR_DATA, IMG_SHEEP)
    #img_sheep = Image.open(fp_sheep)
    #img_list = [img_cow, img_sheep]
    #pred_animals = predict_class(model, img_list, fp_labels)
    #plot_prediction(pred_animals)

    # SUBQUESTION 1.8
    #pred_cow = predict_segment(model, [img_cow])
    #plot_object(pred_cow, 'cow')

    # SUBQUESTION 1.9
    #fp_beach = os.path.join(root, DIR_DATA, IMG_BEACH)
    #img_beach = Image.open(fp_beach) 
    ## we resize the cow to fit in the beach image   
    #img_cow.thumbnail(size = (250, 250))
    #pred_cow = predict_segment(model, [img_cow])
    #masked_cow = create_obj_mask(pred_cow)
#
    #img_beach.paste(img_cow, (300, 150))
    #img_beach.save('mod_beach.jpg')


if __name__ == "__main__":
    main()