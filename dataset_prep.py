import os, shutil
import numpy as np
# import splitfolders
import argparse
parser = argparse.ArgumentParser(description='Data preparation script')
parser.add_argument('--dataset_dir', type=str, help='name of the path where dataset is')
parser.add_argument('--percentage', default=0.3, type=float, help='percentage of supervised data')
parser.add_argument('--split_train_test', action="store_true", help="If true, split the dataset into train test and val folder")

args = parser.parse_args()
dataset_path = "/content/COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset"
#Create pretext and downstream file
def pre_text_dowstream_ds_split(root='', percentage_split=0.3):
  if not os.path.exists("pretext") or not os.path.exists("downstream"):
    os.mkdir(os.path.join("pretext"))
    os.mkdir(os.path.join("downstream"))
  ls=os.listdir(root)
  # print(ls)
  for folder in ls:
    cat=os.listdir(os.path.join(root, folder))
    for classes in cat:
      img=os.listdir(os.path.join(root, folder, classes))
      images_per_class=len(img)
      if not os.path.exists(os.path.join("pretext", folder, classes)):
        os.makedirs(os.path.join("pretext", folder, classes))
      if not os.path.exists(os.path.join("downstream", folder, classes)):
        os.makedirs(os.path.join("downstream", folder, classes))
      image_to_be_moved=np.floor(percentage_split*images_per_class)
      image_to_be_moved=int(image_to_be_moved)
      downstream, pretext = img[:image_to_be_moved], img[image_to_be_moved:]
      # print("Current class:\t", classes)
      # print("\nDownstream:\t",downstream,"\nPretext:\t", pretext)
      # print("\nFile in ds:\t:", len(downstream),
      #       "\nFile in pt:\t:", len(pretext))
      for image in pretext:
        shutil.copyfile(os.path.join(root, folder, classes, image), os.path.join("pretext", folder, classes, image))
        # print("From:\t", os.path.join(root, folder, classes, image), "\tTo:\t", os.path.join("pretext", folder, classes, image))
      for image in downstream:
        shutil.copyfile(os.path.join(root, folder, classes, image), os.path.join("downstream", folder, classes, image))
        # print("From:\t", os.path.join(root, folder, classes, image), "\tTo:\t", os.path.join("downstream", folder, classes, image))
    
def main():
    # if(args.split_train_test):
        # splitfolders.ratio(args.dataset_dir, output=args.dataset_dir+"_split",
        # seed=1337, ratio=(.8,.1, .1), group_prefix=None, move=True) # default values
        # pre_text_dowstream_ds_split(root=args.dataset_dir+"_split", percentage_split=args.percentage)
    pre_text_dowstream_ds_split(root=args.dataset_dir, percentage_split=args.percentage)
if(__name__=="__main__"):
    main()