Fast voice conversion
=================
Branch description: This branch is for pytorch 1.0.1
# 1. Data preparation
* **Under the directory of your data, create 'train', 'eval', 'test' and split your data.**
* Target speaker can have 'train' and 'eval' while source speaker has 'train', 'eval', and 'test'
* 'train2', 'train3' reads data from 'train' and 'eval' while 'convert' and 'quick_convert' reads data from 'test' directory. 
# 2. Directory 
## 2.1 Runnable
* Runnable: package contains python modules that you actually run.
* logdir: According to your specification in cofig.yaml, 'exp_name' option, matching directory is created. For example, if you specify train2.exp_name as 'female_to_male', then 'female_to_male/train2' directory will be created under 'logdir' directory. If you specify the same name for train3.exp_name, 'female_to_male/train3 directory will also be created. 
  * Files that are saved in specified logdir: tensorboard event files, pytorch checkpoint, log file
  * converted wav: If you convert or quick_convert using checkpoint of certain directory, convert_[time_stamp] directory is created under that logdir. For example, if you run convert using checkpoint at 'logdir/female_to_male/train2', then 'convert_[time_stamp]' directory is created under train2. Log file for conversion and generated wav files are saved there.
## 2.2 configs
* This is where your yaml file is stored. you need to specify the name of configuration yaml at each runnables.
# 3. How to train
* train1
  * dadta_path: It assumes data_path for TIMIT
  * exp_name: checkpoint, tensorboard event files, log files will be saved under 'logdir/[exp_name]/train1'
* train2
  * data_path: You have to create 'train' and 'eval' directory and split your DB under the path you specified for 'data_path'.
  * exp_name: checkpoint, tensorboard event files, log files will be saved under 'logdir/[exp_name]/train2'
  * **When you run 'train2', it refers 'train1.exp_name' to search for trained network 1 (phone recognizer).**
* train 3
  * data_path: You have to create 'train' and 'eval' directory and split your DB under the path you specified for 'data_path'.
  * exp_name: checkpoint, tensorboard event files, log files will be saved under 'logdir/[exp_name]/train3'
  * **When you run 'train3', it refers 'train1.exp_name' to search for trained network 1 (phone recognizer) and 'train2.exp_name' to search for trained network 2.**
# 4. How to convert
1. configure.yaml
  * **'convert.py' requires network 1, network 2 and the source speech from which you want to convert to the target speaker. Thus you must specify the following 3 items in config.yaml**
  * convert.data_path: You have to create 'test' directory under the path that you have speicified here. For example, If you specify as below, it loosk for '/home/admin/Music/VCTK/VCTK-Corpus/wav16/p254/test'
     ```
     data_path: '/home/admin/Music/VCTK/VCTK-Corpus/wav16/p254'
     ```
  * train1.exp_name: Write the exp_name where the checkpoint of network 1 is saved.
  * train2.exp_name: Write the exp_name where the checkpoint of network 2 is saved.
2.  Run convert.py
  * converted output wav file: converted outputs are saved at 
  ```
  [exp_name]/train2/convert_[timestamp]
  ```
  if you generated them by running convert.py
# 5. How to 'quick_convert'
1. configure.yaml
  * **'quick_convert.py' requires network 3 and thet source speech from which you want to convert to the target speaker. Thus you must specify the following 2 items in config.yaml**
  * quick_convert.data_path: You have to create 'test' directory under the path that you have speicified here. For example, If you specify as below, it loosk for '/home/admin/Music/VCTK/VCTK-Corpus/wav16/p254/test'
     ```
     data_path: '/home/admin/Music/VCTK/VCTK-Corpus/wav16/p254'
     ```
  * quick_convert.exp_name: Write the exp_name where network3 checkpoint is saved. 
  For example, if it is in 
  ```
  /logdir/female_to_male/train3
  ```
  then write as follows
  ```
  quick_convert:
    exp_name: 'female_to_male'
  ```
2. Run quick_convert.py
  * converted output wav file: converted outputs are saved at 
  ```
  [exp_name]/train3/convert_[timestamp]
  ```
  if you generated them by running quick_convert.py

