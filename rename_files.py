import os 

dir_ = "betterFaster/sim_utils/fake_data"
for file in os.listdir(dir_): 
    if "all_clique_feats" in file: 
        #exp_0_fake_all_clique_feats.pickle -> exp_0_all_clique_feats.pickle 
        i0 = file.index("_fake")
        i1 = file.index("all_clique_feats")
        sub_str = file[:i0].split("_")
        s0 = sub_str[0] + sub_str[1]
        new_filename = s0 + file[i1:]
        #print("new_filename: ",new_filename)
        os.rename(os.path.join(dir_,file),os.path.join(dir_,new_filename)) 
    elif "fake_observations" in file:
        #exp0_fake_observations.pickle -> exp0_observations.pickle 
        i0 = [i for i,x in enumerate(file) if x == "_"][0]
        #i1 = [i for i,x in enumerate(file) if x == "_"][1]
        new_filename = file[:i0] + "observed_cliques.pickle"
        os.rename(os.path.join(dir_,file),os.path.join(dir_,new_filename))