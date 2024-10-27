import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import pickle
device = torch.device('cpu')


class RawProcess():

    def __init__(self, num_runs=45,num_runs_test=5,steps_per_run=250, box_side_length=15.0, N_atoms=2700):

        dim = N_atoms * 6
        self.num_runs = num_runs
        self.num_runs_test = num_runs_test
        self.steps_per_run = steps_per_run
        self.box_side_length = box_side_length
        self.N_atoms = N_atoms 
 
    def read_trajectory(self, filename,records,batchcd ):
        # read raw lammps data
        with open(filename, "r") as f:
            dat = f.read()
        lines = dat.split("\n")

        # Extract the number of particles and the edge length of the simulation box.
        num_particles = int(lines[3])
        box = np.fromstring(lines[5], dtype=np.float32, sep=" ")
        box_length = box[1] - box[0]

        # Iterate over all timesteps and extract the relevant data columns.
        header_size = 9
        record_size = header_size + num_particles
        num_records = len(lines) // record_size -1
        
        try: 
            assert num_records == self.steps_per_run
            for i in range(num_records):
                record = lines[header_size + i * record_size : (i + 1) * record_size]
                record = np.array([l.split(" ") for l in record], dtype=np.float32)
                records[f'step_{i+batch*self.steps_per_run}'] = record[:,2:]

        except:
            print(filename, num_records)
 
        return records, box_length
    
    def fix_displacements(self, train_xyz, idx, period):
        # Since pbc(periodic boundary condition) is used in the lammps data generation, 
        # The x,y,z coordinates of each trajectory maybe discontinuous, we fix the discontinuity here 
        for i in range(train_xyz.shape[1]):

            # Find points with large displacements
            start = self.steps_per_run*idx
            end = self.steps_per_run*(idx+1)
            displacements = np.abs(train_xyz[start+1:end,i] - train_xyz[start:end-1,i])
            large_displacements = np.where(displacements > 5)[0] + self.steps_per_run * idx

            for j in large_displacements:
                # Calculate actual jump size and wrap around period if necessary
                jump_size = train_xyz[j + 1, i] - train_xyz[j, i]
                if abs(jump_size) > period / 2:
                    if jump_size >0:
                        train_xyz[j + 1:end, i] -= period 
                    else:
                        train_xyz[j + 1:end, i] += period 

        return train_xyz    
    
    def process(self, records):
        X = []
        Y = []
        for key in records.keys():
            X.append(records[key][:,:6])
            Y.append(records[key][:,3:])

        X = np.array(X)
        Y = np.array(Y)

        xyz = X[:,:,:3].reshape(X.shape[0],-1)
        vxyz = X[:,:,3:].reshape(X.shape[0],-1)
        fxyz = Y[:,:,3:].reshape(Y.shape[0],-1)

        num_runs = len(records) // self.steps_per_run
        for idx in range(num_runs):
            print(f"Processing training trajectory {idx}")
            xyz = self.fix_displacements(xyz,idx,period=self.box_side_length)

        xyz = torch.tensor(xyz).to(device)
        vxyz = torch.tensor(vxyz).to(device)
        fxyz = torch.tensor(fxyz).to(device)

        X = torch.cat([xyz,vxyz],-1)
        Y = torch.cat([vxyz,fxyz],-1)
        return X, Y

    def forward(self):
        
        # Load raw lammps data  

        # records= {}
        # for batch in tqdm(range(self.num_runs)):
        #     filepath = f'./raw_data/output_run_{batch+1}.xyz'
        #     records,_ = self.read_trajectory(filepath,records,batch)

        # # Read test data 
        # records_test = {}
        # for batch in tqdm(range(self.num_runs, self.num_runs + self.num_runs_test)):
        #     filepath = f'./raw_data/output_run_{batch+1}.xyz'
        #     records_test,_ = self.read_trajectory(filepath,records_test,batch-self.num_runs) 
        # 

        # load saved data 
        with open(f'./raw_data/train_{self.N_atoms}.pkl', 'rb') as f:
            records = pickle.load(f)

        with open(f'./raw_data/test_{self.N_atoms}.pkl', 'rb') as f:
            records_test = pickle.load(f)
   
        print('Number of training data points:', len(records))
        print('Number of test data points:',len(records_test))
        print('Each trajecory is of length:',self.steps_per_run)

        train_X, train_Y = self.process(records)
        test_X, test_Y = self.process(records_test)

        # save data 
        torch.save({'X':train_X, 'Y':train_Y}, f'./processed_data/train_{self.N_atoms}.pt')
        torch.save({'X':test_X, 'Y':test_Y},f'./processed_data/test_{self.N_atoms}.pt')


if __name__ == "__main__":
    Processor = RawProcess()
    Processor.forward()


