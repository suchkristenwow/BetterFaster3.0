import os 

exp_dir = "/media/arpg/easystore/BetterFaster/kitti_carla_simulator/exp_results/"
sub_dirs = os.listdir("/media/arpg/easystore/BetterFaster/kitti_carla_simulator/exp_results/")
for dir_ in sub_dirs:
    path = os.path.join(exp_dir,dir_)
    for file in os.listdir(path): 
        if not os.path.isdir(file):
            for exp in range(12):
                if "experiment1" not in file:
                    try:
                        os.remove(os.path.join(path,file))
                    except:
                        continue 
                '''
                if "experiment" + str(exp) in file:
                    print("removing file: ",file)
                    os.remove(os.path.join(path,file)) 
                '''
        else:
            sub_sub_dir = os.path.join(path,file) 
            for orb_file in os.listdir(sub_sub_dir):    
                for exp in range(12):
                    if "experiment1" not in file:
                        try:
                            os.remove(os.path.join(path,file))
                        except:
                            continue 
                    '''
                    if "experiment" + str(exp) in file:
                        print("removing file: ",file)
                        os.remove(os.path.join(path,file)) 
                    '''
                    