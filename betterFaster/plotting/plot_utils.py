import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
import os 
from betterFaster.sim_utils.utils import get_reinitted_id 


def wrap_angle(angle): 
    # Use modulo operation to wrap the angle
    wrapped_angle = angle % 360
    
    # Ensure the result is positive
    if wrapped_angle < 0:
        wrapped_angle += 360

    return wrapped_angle

class betterFaster_plot:
    def __init__(self,exp,parameters,gt_car_traj,observed_clique_ids,gt_gstates,data_association=None):
        plt.ion() 
        print("initialize plotter...")
        sim_length = parameters["sim_length"]
        results_dir = parameters["results_dir"]
        
        if len(observed_clique_ids) <= 8:
            n_axis = len(observed_clique_ids)
        else:
            n_axis = 8
        self.posterior_cache = []
        self.observations_cache = np.zeros((sim_length,len(observed_clique_ids)))

        #Init all the figure axes 
        fig, ax = plt.subplots() #this is for the animation thing 
        self.fig = fig; self.ax = ax 
        self.traj_err_fig, self.traj_err_ax = plt.subplots() 
        self.posterior_fig, self.posterior_ax = plt.subplots(nrows=1, ncols=n_axis, figsize=(24, 4))
        self.lm_estimate_fig, self.lm_estimate_ax = plt.subplots(nrows=1, ncols=n_axis, figsize=(24,4))
        self.gstate_err_fig, self.gstate_err_ax = plt.subplots(nrows=1, ncols=n_axis,figsize=(24,4)) 

        self.exp = exp 
        self.sim_length = sim_length
        self.observed_clique_ids = observed_clique_ids
        self.fov = parameters["vehicle_parameters"]["fov"]
        self.gt_gstates = gt_gstates 
        #extract ground truth landmarks 
        cone_ids_file = os.path.join(results_dir,"cone_ids/experiment"+str(self.exp + 1)+"cone_ids_.txt")
        # Read the contents of the text file
        with open(cone_ids_file, 'r') as file:
            lines = file.readlines()
        cone_ids = np.unique([int(line.strip()) for line in lines])

        # Convert each line to an integer and store in a list
        tree_ids_file = os.path.join(results_dir,"tree_ids/experiment"+str(self.exp + 1)+"tree_ids_.txt")
        # Read the contents of the text file
        with open(tree_ids_file, 'r') as file:
            lines = file.readlines()
        # Convert each line to an integer and store in a list
        tree_ids = np.unique([int(line.strip()) for line in lines]) 

        self.gt_trees = []
        if data_association is None:
            data_association_path = os.path.join(results_dir,"data_association/experiment"+str(self.exp + 1)+"data_association.csv")
            if not os.path.exists(data_association_path):
                data_association_path = os.path.join(results_dir,"data_associations/exp"+str(self.exp)+"_data_association.csv") 
            data_association = np.genfromtxt(data_association_path,delimiter=" ") 
        self.data_association = data_association 

        all_data_associations = {} 
        try:
            data_assocation_files = os.listdir(os.path.join(results_dir,"data_association"))  
        except:
            data_assocation_files = os.listdir(os.path.join(results_dir,"data_associations"))  

        for file in data_assocation_files:
            #print("this is file: ",file)
            try:
                idx0 = 10; idx1 = -20 
                #print("file[idx0:idx1]: ",file[idx0:idx1])
                exp = int(file[idx0:idx1])
            except:
                idx0 = 3; idx1 = -21
                #print("file[idx0:idx1]: ",file[idx0:idx1])
                exp = int(file[idx0:idx1])
            #print("this is exp: ",exp)
            try:
                all_data_associations[exp] = np.genfromtxt(os.path.join(results_dir,"data_association/" + file),delimiter=" ")
                #print("this is data_association path: ",os.path.join(results_dir,"data_association/" + file))
            except: 
                #print("this is the exception")
                data_association_path = os.path.join(results_dir,"data_associations/exp"+str(exp)+"_data_association.csv") 
                #print("this is data_association_path:",data_association_path )
                #print("data association result: ",np.genfromtxt(data_association_path,delimiter=" "))
                all_data_associations[exp+1] = np.genfromtxt(data_association_path,delimiter=" ") 

        if not np.array_equal(all_data_associations[self.exp + 1],data_association):
            print("all_data_associations[self.exp]: ",all_data_associations[self.exp + 1])
            print("data_association:",data_association)
            print("im confusion")
            raise OSError
        
        self.all_data_associations = all_data_associations
        '''
        for x in data_association[:,0]:
            print("this is x:",x)
            if not x in all_data_associations[self.exp][:,0]:
                raise OSError 
            else:
                idx = np.where(all_data_associations[self.exp][:,0] == x)
                #print("all_data_associations[n][idx,:]: ",all_data_associations[self.exp][idx,:])
            print("self.exp: ",self.exp)
            x = get_reinitted_id(all_data_associations,self.exp,x)
            print("reinitted id: ",x)
        raise OSError
        '''
        #self.gt_trees = [x for x in data_association if x[0] in self.observed_clique_ids and get_reinitted_id(all_data_associations,self.exp,x[0]) in tree_ids]
        #self.gt_cones = [x for x in data_association if x[0] in self.observed_clique_ids and get_reinitted_id(all_data_associations,self.exp,x[0]) in cone_ids]
        self.gt_trees = [x for x in data_association if x[0] in tree_ids] 
        self.gt_cones = [x for x in data_association if x[0] in cone_ids]
        self.gt_cones = np.array(self.gt_cones)
        self.gt_trees = np.array(self.gt_trees)
        if len(self.gt_trees) == 0 and len(self.gt_cones) == 0:
            print("there are no trees or cones!!!")
            print("self.observed_clique_ids:",self.observed_clique_ids)
            print("data_association:",data_association) 
            print("cone_ids: {},tree_ids: {}".format(cone_ids,tree_ids))
            raise OSError 
        self.ax_bounds = self.determine_plot_bounds(gt_car_traj,observed_clique_ids)
        self.all_observed_cliques = []
        if len(self.gt_trees) == 0:
            raise OSError 
        #extract ground truth trajectory 
        self.gt_traj = gt_car_traj 
        self.traj_estimate_cache = np.zeros((sim_length,3))
        self.sim_length = sim_length 

        self.lm_estimate_err_cache = {} 
        for id_ in data_association[:,0]:
            self.lm_estimate_err_cache[id_] = np.zeros((sim_length,))    

    def plot_state(self,slam,t,robot_pose, observations_t, posteriors, growth_state_estimates):
        print("entering plot state...")
        print("this is robot_pose: ",robot_pose)
        self.ax.clear()
        self.ax.set_xlim(self.ax_bounds[0],self.ax_bounds[1])
        self.ax.set_ylim(self.ax_bounds[2],self.ax_bounds[3])
        self.ax.set_aspect('equal')
        #plot the car 
        self.ax.scatter(robot_pose[0],robot_pose[1],color="k",s=5)

        observed_clique_ids_t = np.array(np.unique([int(x["clique_id"]) for x in observations_t]))
        self.all_observed_cliques.extend([x for x in observed_clique_ids_t if x not in self.all_observed_cliques])
        
        #print("this is observed clique ids at this timestep: ",observed_clique_ids_t)
        for clique_id in observed_clique_ids_t:
            #print("updating observations_cache for clique_id: ",clique_id)
            idx = self.observed_clique_ids.index(clique_id)
            #print("this is idx:",idx)
            if not isinstance(idx,int):
                raise OSError
            self.observations_cache[t,idx] = 1
            '''
            print("this is t: ",t)
            print("self.observations_cache[t,idx]: ",self.observations_cache[t,idx])
            '''

        #plot the frustrum
        self.plot_frustrum(robot_pose)
    
        #plot the gt trees 
        if len(self.gt_trees) > 0:
            '''
            print("scattering trees... ")
            print("there are {} trees".format(len(self.gt_trees)))
            print("self.gt_trees: ",self.gt_trees)
            '''
            for i in range(len(self.gt_trees)): 
                #if self.gt_trees[i,0] in self.observed_clique_ids:
                self.ax.scatter(self.gt_trees[i,1],self.gt_trees[i,2],color='green', marker='*')

            for i in range(len(self.gt_trees)):
                #all_data_associations,n,id_)
                reinitted_id = get_reinitted_id(self.all_data_associations,self.exp,self.gt_trees[i,0])
                self.ax.text(self.gt_trees[i,1],self.gt_trees[i,2],str(reinitted_id),fontsize=8,color="k",ha='center', va='center')

        else:
            print("WARNING no trees")
        
        #plot the gt cones 
        if len(self.gt_cones) > 0:
            '''
            print("scattering cones... ")
            print("there are {} cones".format(len(self.gt_trees)))
            print("self.gt_cones: ",self.gt_trees)
            '''
            for i in range(len(self.gt_cones)):
                #if get_reinitted_id(self.gt_cones[i,0]) in self.observed_clique_ids:
                self.ax.scatter(self.gt_cones[i,1],self.gt_cones[i,2],color="orange",marker="^")

            for i in range(len(self.gt_cones)):
                #all_data_associations,n,id_)
                reinitted_id = get_reinitted_id(self.all_data_associations,self.exp,self.gt_cones[i,0]) 
                self.ax.text(self.gt_cones[i,1],self.gt_cones[i,2],str(reinitted_id),fontsize=8,color="k",ha='center', va='center')
            
        else:
            print("WARNING no cones ")

        #plot the ground truth trajectory 
        if t > 0:
            self.ax.plot(self.gt_traj[:t,0],self.gt_traj[:t,1],'k',alpha=0.5)

        #plot the estimated trajectory 
        x = robot_pose[0]
        y = robot_pose[1]

        print("robot_pose: ",robot_pose)
        yaw = robot_pose[-1]
        print("yaw:",yaw)

        #robot_circle = plt.Circle((x, y), 1.5, color='red', fill=True)
        robot_circle = plt.Circle((x, y), 0.35, color='red', fill=True)
        # Calculate the end point of the arrow based on x, y, and yaw
        dx = 1 * np.cos(yaw)
        dy = 1 * np.sin(yaw)
        #robot_pointer = self.ax.arrow(x, y, dx, dy, head_width=2, head_length=2, fc='red', ec='red')
        robot_pointer = self.ax.arrow(x, y, dx, dy, head_width=0.5, head_length=0.5, fc='red', ec='red')
        self.ax.add_patch(robot_circle)
        self.ax.add_patch(robot_pointer)

        self.traj_estimate_cache[t,:] = np.array([x,y,yaw])
        #plot the best estimate trajectory
        if t > 0:
            self.ax.plot(self.traj_estimate_cache[1:t,0],self.traj_estimate_cache[1:t,1],"r:")

        self.traj_err_cache = np.zeros((self.sim_length,))

        #plot the observations 
        self.plot_observations(robot_pose,observations_t) 

        self.plot_lm_ellipsoids(slam)

        self.plot_posteriors(t,posteriors,observations_t)

        self.plot_lm_estimate_err(slam,observations_t,t) 

        self.plot_gstate_err(t,growth_state_estimates)

        self.plot_trajectory_err(t,robot_pose)

        plt.pause(0.05)

        if t == self.sim_length - 1:
            if not os.path.exists("debugPlots"): 
                os.mkdir("debugPlots")
            plt.close(self.fig) 
            self.posterior_fig.savefig(f"debugPlots/exp{self.exp}_posteriors.jpg")
            plt.close(self.posterior_fig) 
            self.lm_estimate_fig.savefig(f"debugPlots/exp{self.exp}_lmEstimateErr.jpg")
            plt.close(self.lm_estimate_fig)
            self.gstate_err_fig.savefig(f"debugPlots/exp{self.exp}_gstateErr.jpg")
            plt.close(self.gstate_err_fig)
            self.traj_err_fig.savefig(f"debugPlots/exp{self.exp}_trajErr.jpg")
            plt.close(self.traj_err_fig)
    
    def plot_trajectory_err(self,t,robot_pose): 
        #print("self.gt_traj[0,:]: ",self.gt_traj[0,:])
        err_t = np.linalg.norm(self.gt_traj[t,:2] - robot_pose[:2])
        self.traj_estimate_cache[t] = err_t  
        self.traj_err_ax.clear()
        self.traj_err_ax.plot(np.arange(t),self.traj_estimate_cache[:t],color="r")
        self.traj_err_ax.set_title("Robot Pose Error")

    def plot_gstate_err(self,t,growth_state_estimates): 
        global_t = (self.exp)*self.sim_length + t   

        #for i,id_ in enumerate(self.gt_gstates[:,0]): 
        for i in range(len(self.gstate_err_ax)):
            print("growth_state_estimates: ",growth_state_estimates)
            #id_ = growth_state_estimates[i,0]
            id_ = list(growth_state_estimates.keys())[i]
            print("id_:",id_)
            self.gstate_err_ax[i].clear() 
            #plot gt for comparison
            if id_ in self.gt_gstates[:,0]: 
                idx = np.where(self.gt_gstates[:,0] == id_)
                gt_gstate_id = int(self.gt_gstates[idx,1]) 
            else: 
                reinit_id = get_reinitted_id(self.all_data_associations,self.exp,id_,optional_exp=self.exp)
                if reinit_id not in self.gt_gstates[:,0]:
                    print("reinit_id: {}, id_:{}".format(reinit_id,id_))
                    print("this is exp: ",self.exp)
                    print("gt gstates: ",self.gt_gstates)
                    raise OSError 
                idx = np.where(self.gt_gstates[:,0] == reinit_id)
                gt_gstate_id = int(self.gt_gstates[idx,1])
            gt_arr = np.ones((global_t,)) * gt_gstate_id 
            self.gstate_err_ax[i].plot(np.arange(global_t),gt_arr,color="k")
            print("growth_state_estimates[id_]:",growth_state_estimates[id_])
            self.gstate_err_ax[i].plot(np.arange(global_t),growth_state_estimates[id_][:global_t],color="r",linestyle="--")
            self.gstate_err_ax[i].set_title(f"Clique {id_}")
            self.gstate_err_ax[i].set_ylim(-0.05,3.05)

        #self.gstate_err_ax.set_title("Mean Growth State Error Rate")
        self.gstate_err_fig.suptitle("Growth State Estimates")

    def plot_lm_estimate_err(self,slam,observations_t,t): 
        observed_cliques_t = np.unique([x["clique_id"] for x in observations_t]) 
        #print("observed_cliques_t: ",observed_cliques_t)
        unobserved_clique_ids = [x for x in self.observed_clique_ids if x not in observed_cliques_t]
        #print("unobserved clique ids: ",unobserved_clique_ids)

        idx = np.argmax([x.weight for x in slam.particles]) 
        best_landmarks = slam.particles[idx].landmark 
        print("plotting lm ellipsoids... this is exp:",self.exp)
        for lm in best_landmarks:
            if not lm.lm_id in self.lm_estimate_err_cache.keys():
                print("self.lm_estimate_err_cache.keys(): ",self.lm_estimate_err_cache.keys())
                print("this lm id is not in the lm estimate err cache:",lm.lm_id)
                id_ = get_reinitted_id(self.all_data_associations,self.exp,lm.lm_id,optional_exp=self.exp)
                print("tried to get the reinitted id: ",id_)
            else:
                id_ = lm.lm_id
            self.lm_estimate_err_cache[id_][t] = self.get_lm_estimate_error(id_,best_landmarks)
            print("updated lm_estimate_err_cache for id_: {}, this is the lm_estimate_err: {}".format(id_,self.lm_estimate_err_cache[id_][t]))

        global_t = (self.exp)*self.sim_length + t 

        for i, ax in enumerate(self.lm_estimate_ax):
            ax.clear()
            inc = self.sim_length/5
            x_bound = min(int(np.ceil(t/inc)*inc) + 10, self.sim_length) 
            ax.set_xlim(0,x_bound)
            if i < len(observed_cliques_t): 
                id_ = observed_cliques_t[i]
                #if id_ in self.all_observed_cliques: 
                if not id_ in self.lm_estimate_err_cache.keys():
                    id_ = get_reinitted_id(self.all_data_associations,self.exp,id_,optional_exp=self.exp)
                err_id = self.lm_estimate_err_cache[id_][:global_t]
                ax.plot(np.arange(global_t),err_id)
            else: 
                #pick random ids 
                id_ = unobserved_clique_ids[i - len(observed_cliques_t)]
                #if id_ in self.all_observed_cliques:
                if not id_ in self.lm_estimate_err_cache.keys():
                    id_ = get_reinitted_id(self.all_data_associations,self.exp,id_,optional_exp=self.exp)
                err_id = self.lm_estimate_err_cache[id_][:global_t] 
                ax.plot(np.arange(global_t),err_id)
            ax.set_ylim(-0.05,12)
            ax.set_title("LM Estimation Error for " + str(id_))
    
    def get_lm_estimate_error(self,id_,best_landmarks): 
        best_landmark_ids = [x.lm_id for x in best_landmarks]
        if not id_ in best_landmark_ids:
            id_ = get_reinitted_id(self.all_data_associations,self.exp,id_,optional_exp=1) 
            if id_ not in best_landmark_ids: 
                raise OSError
            
        idx = best_landmark_ids.index(id_)
        center = best_landmarks[idx].EKF.mu

        if id_ in self.gt_trees[:,0]:
            idx = np.where(self.gt_trees[:,0] == id_)
            gt_loc = self.gt_trees[idx,1:]
            #print("this is a tree: ",gt_loc)
        elif id_ in self.gt_cones[:,0]: 
            idx =  np.where(self.gt_cones[:,0] == id_)
            gt_loc = self.gt_cones[idx,1:]
            #print("this is a cone: ",gt_loc)
        else:
            print("couldnt find this id_.... ",id_)
            id_ = get_reinitted_id(self.all_data_associations,self.exp,id_,optional_exp=self.exp)
            print("this is the reinitted_id: ",id_)
            if id_ in self.gt_trees[:,0]: 
                idx = np.where(self.gt_trees[:,0] == id_)
                gt_loc = self.gt_trees[idx,1:]
            elif id_ in self.gt_cones[:,0]:  
                idx = np.where(self.gt_cones[:,0] == id_)
                gt_loc = self.gt_cones[idx,1:]
            '''
            else:
                gt_loc = None 
                random_id = self.gt_cones[0]
                for k in self.all_data_associations.keys():
                    if random_id in self.all_data_associations[k][:,0]:
                        #TO DO 
            '''
                        
        if len(gt_loc) == 0:
            id_ = get_reinitted_id(self.all_data_associations,self.exp,id_,optional_exp=self.exp)
            if id_ in self.gt_cones[:,0]: 
                idx = np.where(self.gt_cones[:,0] == id_)
                gt_loc = self.gt_cones[idx,1:]
            elif id_ in self.gt_trees[:,0]: 
                idx = np.where(self.gt_trees[:,0] == id_)
                gt_loc = self.gt_cones[idx,1:]
            else: 
                raise OSError 
        
        print("gt_loc: ",gt_loc)
        gt_loc = np.reshape(gt_loc,(2,1))
        print("this is center: ",center)
        err_ = np.linalg.norm(gt_loc - center) 
        print("err_ :",err_)

        if np.isnan(err_) or np.isinf(err_): 
            raise OSError 
        
        return err_ 
        
    def plot_lm_ellipsoids(self,slam): 
        idx = np.argmax([x.weight for x in slam.particles]) 
        best_landmarks = slam.particles[idx].landmark 
        print("plotting lm ellipsoids... this is exp:",self.exp)
        for lm in best_landmarks:
            lm_EKF = lm.EKF 
            center = lm_EKF.mu  
            if not np.all(center == 0):
                #print("lm_id: {}, center_estimate: {}".format(lm.lm_id,center))
                self.ax.scatter(center[0],center[1],color="blue",s=2)
                self.ax.text(center[0] - 0.1,center[1] + 0.1,str(lm.lm_id),fontsize=8,color="blue",ha='center',va='center')
                eigenvalues,eigenvectors = np.linalg.eigh(lm_EKF.Sigma) 
                angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1])) 
                # Create the ellipse
                ellipse = Ellipse(xy=center, width=2*np.sqrt(eigenvalues[0]), height=2*np.sqrt(eigenvalues[1]),angle=angle,edgecolor="k",facecolor="None")
                self.ax.add_patch(ellipse) 

    def plot_posteriors(self,t,posteriors_t,observations_t):
        #self.posterior_fig, self.posterior_ax
        #print("plotting posteriors...")
        tmp = []
        for c in sorted(posteriors_t.keys()):
            global_t = (self.exp)*self.sim_length + t 
            tmp.append(posteriors_t[c][global_t])

        print("this is tmp:",tmp)
        self.posterior_cache.append(tmp)

        if t ==0:
            return 
        
        plot_cache = np.array(self.posterior_cache)

        observed_cliques_t = np.unique([x["clique_id"] for x in observations_t]) 
        print("observed_cliques_t: ",observed_cliques_t)
        unobserved_clique_ids = [x for x in self.observed_clique_ids if x not in observed_cliques_t]
        print("unobserved_clique_ids:",unobserved_clique_ids)

        # Plot something in each subplot
        for i, ax in enumerate(self.posterior_ax):
            ax.clear()
            inc = self.sim_length/5
            x_bound = np.ceil(t/inc)*inc 
            ax.set_xlim(0,x_bound)
            ax.set_ylim(-0.05,1.05)
            if i < len(observed_cliques_t):
                clique_id = observed_cliques_t[i] 
            else: 
                clique_id = unobserved_clique_ids[i - len(observed_cliques_t)]
            idx = sorted(posteriors_t.keys()).index(clique_id) 
            #print("idx:",idx)
            ax.set_title("Posteriors for Clique " + str(clique_id))
            if len(np.arange(t+1)) != len(plot_cache[:,idx]):
                #print("len(plot_cache[:idx]): ",len(plot_cache[:idx])) 
                t_prime = t + 1 - len(plot_cache[:,idx])
                #print("len(np.arange(t_prime,t+1)): ",len(np.arange(t_prime,t+1))) 
                #print("this is t: {}, and this is t_prime: {}".format(t,t_prime))
                ax.plot(np.arange(t_prime,t+1),plot_cache[:,idx],color="k")
                #print("plot_cache[:,idx]:",plot_cache[:,idx])
                ax.scatter(t,plot_cache[-1,idx],color="r",marker="*")
            else: 
                ax.plot(np.arange(t+1),plot_cache[:,idx],color="k")
                ax.scatter(t,plot_cache[t,idx],color="r",marker="*")
                '''
                print("this is t: ",t)
                print("plot_cache: ",plot_cache)
                print("plot_cache[t,idx]: ",plot_cache[t,idx])
                '''
                
            #print("plot_cache[t,idx]:",plot_cache[t,idx])
            ax.plot(np.arange(self.sim_length),np.ones((self.sim_length,))*0.9,color="red",linestyle="--")
            ax.plot(np.arange(self.sim_length),np.ones((self.sim_length,))*0.5,color="blue",linestyle="--")

            idx = self.observed_clique_ids.index(clique_id)
            landmark_observed = self.observations_cache[:,idx]

            start_idx = None
            for j, observed in enumerate(landmark_observed):
                '''
                if j == t:
                    #print("at this time this is observed for this landmark: ",landmark_observed[t])
                '''
                if observed == 1 and start_idx is None:
                    start_idx = j 
                    #print("updated start idx to: ",start_idx)
                elif observed == 0 and start_idx is not None:
                    #print("start_idx is not none: ",start_idx)
                    rect = plt.Rectangle((start_idx, 0), j - start_idx, 1, alpha=0.25, color='blue')
                    ax.add_patch(rect)
                    start_idx = None

    def plot_observations(self,robot_pose,observations_t): 
        #self.ax_bounds = (x_lower_bound,x_upper_bound,y_lower_bound,y_upper_bound)
        observed_clique_ids = np.unique([x["clique_id"] for x in observations_t]) 

        for id_ in observed_clique_ids: 
            observations_id = [x for x in observations_t if x["clique_id"] == id_]
            observation_ranges = []; observation_bearings = []
            for obs in observations_id:
                observation_ranges.append(obs["range"])
                #observation_bearings.append(obs["bearing"]*(np.pi/180)) 
                observation_bearings.append(wrap_angle(obs["bearing"])) 
            mean_range = np.mean(observation_ranges); mean_bearing = np.mean(observation_bearings) * (np.pi/180)

            observation_x = robot_pose[0] + mean_range*np.cos(mean_bearing)
            observation_y = robot_pose[1] + mean_range*np.sin(mean_bearing)
            
            print("mean_obervation estimate for clique {}: {},{}".format(id_,observation_x,observation_y))

            if id_ in self.gt_trees[:,0]: 
                self.ax.plot([robot_pose[0],observation_x],[robot_pose[1],observation_y],linestyle="-.",color="green")
                self.ax.scatter(observation_x,observation_y,color="green",marker="x")    
            else:
                self.ax.plot([robot_pose[0],observation_x],[robot_pose[1],observation_y],linestyle="-.",color="orange")
                self.ax.scatter(observation_x,observation_y,color="orange",marker="x")

    def plot_frustrum(self,robot_pose): 
        min_d = 0.1; max_d = 10
        #min_theta = np.deg2rad(robot_pose[-1]*(180/np.pi) - (self.fov/2))
        #max_theta = np.deg2rad(robot_pose[-1]*(180/np.pi)  + (self.fov/2))
        min_theta = robot_pose[-1] - np.deg2rad(self.fov/2)
        max_theta = robot_pose[-1] + np.deg2rad(self.fov/2)
        x1 = robot_pose[0] + min_d*np.cos(min_theta)
        y1 = robot_pose[1] + min_d*np.sin(min_theta)
        x0 = robot_pose[0] + min_d*np.cos(max_theta)
        y0 = robot_pose[1] + min_d*np.sin(max_theta)
        x2 = robot_pose[0] + max_d*np.cos(min_theta)
        y2 = robot_pose[1] + max_d*np.sin(min_theta)
        x3 = robot_pose[0] + max_d*np.cos(max_theta)
        y3 = robot_pose[1] + max_d*np.sin(max_theta)
        self.ax.plot([x0,x1],[y0,y1],'b')
        self.ax.plot([x1,x2],[y1,y2],'b')
        self.ax.plot([x2,x3],[y2,y3],'b')
        self.ax.plot([x3,x0],[y3,y0],'b')

    def determine_plot_bounds(self,gt_car_traj,observed_clique_ids):
        #determine plot bounds 
        xmin = min(gt_car_traj[:,0]); xmax = max(gt_car_traj[:,0])
        ymin = min(gt_car_traj[:,1]); ymax = max(gt_car_traj[:,1])
        #print("self.gt_cones:",self.gt_cones)
        #print("self.gt_trees: ",self.gt_trees)
        landmark_xmin = min([min(self.gt_cones[:,0]),min(self.gt_trees[:,0])])
        landmark_xmax = max([max(self.gt_cones[:,0]),max(self.gt_trees[:,0])])
        delta_x = landmark_xmax - landmark_xmin 
        landmark_ymin = min([min(self.gt_cones[:,1]),min(self.gt_trees[:,1])])
        landmark_ymax = max([max(self.gt_cones[:,1]),max(self.gt_trees[:,1])])
        delta_y = landmark_ymax - landmark_ymin 

        margin = 0.25
        x_lower_bound = xmin - np.abs(xmin)*margin; x_upper_bound = xmax + np.abs(xmax)*margin  
        if min([x_lower_bound,landmark_xmin]) == landmark_xmin: 
            x_lower_bound = landmark_xmin - delta_x*margin 
        if max([x_upper_bound,landmark_xmax]) == landmark_xmax: 
            x_upper_bound = landmark_xmax + delta_x*margin  

        y_lower_bound = ymin - np.abs(ymin)*margin; y_upper_bound = ymax + np.abs(ymax)*margin 
        if min([y_lower_bound,landmark_ymin]) == landmark_ymin: 
            y_lower_bound = landmark_ymin - delta_y*margin 
        if max([y_upper_bound,landmark_ymax]) == landmark_ymax: 
            y_upper_bound = y_upper_bound + delta_y*margin 

        delta_x = x_upper_bound - x_lower_bound
        delta_y = y_upper_bound - y_lower_bound
        
        if max([delta_x,delta_y]) == delta_x:
            #print("delta_x is greater...")
            m = np.mean([y_lower_bound,y_upper_bound])
            #print("m: ",m)
            y_lower_bound = m - np.abs(delta_x/2); y_upper_bound = m + np.abs(delta_x/2)
        else:
            #print("delta_y is greater...")
            m = np.mean([x_lower_bound,x_upper_bound])
            #print("m: ",m)
            x_lower_bound = m - np.abs(delta_y/2); x_upper_bound = m + np.abs(delta_y/2)

        delta_x = x_upper_bound - x_lower_bound
        delta_y = y_upper_bound - y_lower_bound

        if np.round(delta_x) != np.round(delta_y):
            raise OSError 

        ax_bounds = (x_lower_bound,x_upper_bound,y_lower_bound,y_upper_bound)
        
        for clique_id in observed_clique_ids:
            idx = np.where(self.data_association[:,0] == clique_id)
            if not x_lower_bound <= self.data_association[idx,1] <= x_upper_bound: 
                if self.data_association[idx,1] < x_lower_bound:
                    x_lower_bound = self.data_association[idx,1] - 1
                elif x_upper_bound < self.data_association[idx,1]: 
                    x_upper_bound = self.data_association[idx,1] + 1 
            if not y_lower_bound <= self.data_association[idx,2] <= y_upper_bound: 
                if self.data_association[idx,2] < y_lower_bound:
                    y_lower_bound = self.data_association[idx,2] - 1 
                elif y_upper_bound < self.data_association[idx,2]: 
                    y_upper_bound = self.data_association[idx,2] + 1 

        return ax_bounds