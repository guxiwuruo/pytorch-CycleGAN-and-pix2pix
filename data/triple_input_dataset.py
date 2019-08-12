import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import random


class tripleinputdataset(BaseDataset):
    """
    This dataset class can load grocery datasets.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        #self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        #self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.GroceryPath=os.path.join(opt.dataroot,opt.phase+'_grocery')
        self.DomainPath=os.path.join(opt.dataroot,opt.phase+'_domain')
        self.filelist = os.listdir(self.GroceryPath)
        label_set = set()
        for file in self.filelist:
            label, data = file.split('@')
            label = int(label)
            label_set.add(label)

        self.label_set = label_set

        label_data = []
        for file in self.filelist:
            file_label = int(file.split('@')[0])
            find = False
            for label, data in label_data:
                if file_label == label:
                    data.append(file)
                    find = True
                    break
            if find:
                continue

            tmp = []
            tmp.append(file)
            label_data.append((file_label, tmp))


        self.label_data = label_data
        self.transform=get_transform(self.opt,grayscale=False)
        #for get_transform  tomorrow

        '''
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        '''

    def __getitem__(self,index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        '''
                A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        C_img_index=set([i for i in range(self.A_size)])-{index}

        # to get C index (not similare with A)
        index_C=random.sample(C_img_index,1)
        C_path = self.A_paths[index_C[0]]
        C_img = Image.open(C_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        # test data input
        C = self.transform_A(A_img)


        return {'A': A, 'B': B, 'C':C, 'A_paths': A_path, 'B_paths': B_path,'C_paths':A_path}
        '''
        #print(index)
        #index = index % len(self.label_set)
        '''
                if index  in self.label_set:
            pass
        else:
            print('index ')
            print(self.label_set)
            index =random.sample(self.label_set, 1)
        '''

        '''
        for label, data in self.label_data:
            label = int(label)
            if index == label:
                positive_name = np.random.choice(data)
                negative_label = random.sample(self.label_set - set([label]),1)
                print('neg_random')
                break

        for label, data in self.label_data:
            print('neg ')
            print(negative_label[0])
            if label == negative_label[0]:
                negative_name = np.random.choice(data)
                break
        '''
        pos_neg=random.sample(self.label_data,2)
        positive_name=random.sample(pos_neg[0][1],1)
        negative_name=random.sample(pos_neg[1][1],1)
        positive=Image.open(os.path.join(self.GroceryPath, positive_name[0])).convert('RGB')
        negative=Image.open(os.path.join(self.GroceryPath, negative_name[0])).convert('RGB')
        domain=Image.open(os.path.join(self.DomainPath,np.random.choice(os.listdir(self.DomainPath)))).convert('RGB')

        # transform
        neg = self.transform(negative)
        pos = self.transform(positive)
        dom = self.transform(domain)

        return {'pos':pos, 'neg':neg, 'dom':dom,
                'pos_path':os.path.join(self.GroceryPath, positive_name[0]),
                'neg_path':os.path.join(self.GroceryPath, negative_name[0]),
                'dom_path':os.path.join(self.DomainPath,np.random.choice(os.listdir(self.DomainPath)))}



    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return len(self.filelist)

