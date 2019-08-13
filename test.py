"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

# 20190812 debug intermediate layer
import torch.nn as nn
import torch

# 20190812 for l2 norm
import torch.nn.functional as F

# 20190812 for plot
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics     #calculate auc

# 20190813 debug the transform for the input data
import torchvision.transforms as transforms



if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.

    #20190812 create matrix to save feature
    #feature_matrix=torch.empty(dataset.__len__(),512,dtype=torch.float16)

    # for debug
    # the number of the feature
    num_feature = dataset.__len__()
    feature_matrix = torch.empty(num_feature, 512, dtype=torch.float32)
    label = []
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):

        # 20190813 debug for the transform used for the input data
        #a= transforms.ToPILImage(data['pos'].squeeze(0))
        #plt.imshow(transforms.ToPILImage(data['pos'].squeeze(0)))


        #if i >= opt.num_test:  # only apply our model to opt.num_test images.

        # debug feature matrix (small)
        if i >= num_feature:
            break
        model.set_input(data)  # unpack data from data loader

        # 20190812 to show data
        # data['pos'].data

        # 20190812 debug for the intermediate layer output
        '''
        test_model_output2=nn.Sequential(*list((model.netvgg16_features_512.children()))[0][0][:-2])(data['pos'].to(model.device))
        test_model_output3 = nn.Sequential(*list((model.netvgg16_features_512.children()))[0][0][:-3])(
            data['pos'].to(model.device))
        test_model_output4 = nn.Sequential(*list((model.netvgg16_features_512.children()))[0][0][:-4])(
            data['pos'].to(model.device))
        test_model_output5 = nn.Sequential(*list((model.netvgg16_features_512.children()))[0][0][:-5])(
            data['pos'].to(model.device))
        test_model_output6 = nn.Sequential(*list((model.netvgg16_features_512.children()))[0][0][:-6])(
            data['pos'].to(model.device))
        '''

        model.test() # run inference
        feature = model.netvgg16_features_512(data['pos'])   # to get object feature

        # 20190812 to save 1-d feature and the corresponding label
        # to determine whether <1 use 'data.numpy()>1'
        feature_matrix[i] = F.normalize(feature,p=2).view(-1).data #save feature
        label.append(data['pos_path'][0].split('/')[-1].split('@')[0])   #save label

        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML

    feature_distance=1-torch.mm(feature_matrix, feature_matrix.t()) #cosine distance

    tpr = []    #true positive rate
    fpr = []    #false positive rate
    all_negative = 0
    all_positive = 0



    threshold_range = np.arange(0, 1.01, 0.01)

    for i in range(num_feature):
        for j in range(i):
            if (label[i]==label[j]):
                all_positive+=1
            else:
                all_negative+=1


    for threshold_index, threshold in enumerate(threshold_range):

        true_positive=0

        false_positive=0


        for i in range(num_feature):
            for j in range(i):
                if label[i] == label[j]:  # if actual labels are same
                    if(feature_distance[i][j]<threshold):     # recognized as positive (same class)
                        true_positive += 1
                    # all_positive += 1

                else: # if actual labels are different
                    if(feature_distance[i][j]<threshold):     #recognized as positive (same class)
                        false_positive +=1
                    # all_negative +=1

        tpr.append(true_positive/all_positive)
        fpr.append(false_positive/all_negative)

    # 20190813 to plot the auc
    plt.figure()
    plt.plot(fpr,tpr,color='darkorange',
             lw=2,label='ROC curve (area=%0.2f)' %metrics.auc(fpr,tpr))
    plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for grocery recognition')
    plt.legend(loc="lower right")
    plt.show()



