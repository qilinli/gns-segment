import os
import glob
import json
import random
import argparse
import numpy as np

def process_data(dataset, step_size, total_step, in_dir, out_dir):
    val_set = ['60-120', '80-140', '100-180']
    test_set = ['60-130', '80-150', '100-170']
    
    # Create out_dir if it does not exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Normalisation parameters
    pos_max, pos_min = np.array([100, 50]), np.array([-2.5, -50])
    strain_mean, strain_std = 150.25897834554772, 83.50737010164767  # Step 2
    
    simulations = glob.glob(in_dir + '*.npz')
    random.shuffle(simulations)
    simulations.sort()
    
    ds_train, ds_valid, ds_test = {}, {}, {}
    train_info, valid_info, test_info = [], [], []
    vels = np.array([]).reshape(0, 2)
    accs = np.array([]).reshape(0, 2)
    
    for idx, simulation in enumerate(simulations):
        print(f"{idx} Reading {simulation}...")
        
        trajectory_name = os.path.basename(simulation)
        
        # Load data from npz
        with np.load(simulation) as data:
            positions = data['positions']
            particle_types = data['particle_types']
            strains = data['strains']
        
        # Identify the time instance when structure response initializes
        mean_strain = strains.mean(axis=1)
        first_nonzero_step_idx = next((i for i, x in enumerate(mean_strain) if x), None)
        
        # Preprocessing positions
        positions = positions[first_nonzero_step_idx-1:first_nonzero_step_idx-1+total_step:step_size,:-4:1,:]
        positions = (positions - pos_min) / (pos_max - pos_min)  # Normalize
        
        # Change the last 4 particles to boundary particles
        particle_types = particle_types[:-4]

        # Preprocessing strains
        strains = strains[first_nonzero_step_idx-1:first_nonzero_step_idx-1+total_step:step_size, :-4:1]
        strains = (strains - strain_mean) / strain_std  # Standardize
        
        print(f"Position shape:{positions.shape}, type shape:{particle_types.shape}, strain shape:{strains.shape}")
        
        # Data splits: train, valid, test
        key = trajectory_name
        trajectory_data = (positions, particle_types, strains)
        
        if any(name in trajectory_name for name in val_set):
            ds_valid[key] = trajectory_data
            valid_info.append(trajectory_name)
        elif any(name in trajectory_name for name in test_set):
            ds_test[key] = trajectory_data
            test_info.append(trajectory_name)
        else:
            ds_train[key] = trajectory_data
            train_info.append(trajectory_name)
    
        # Extract Vel and Acc statistics
        # positions of shape [timestep, particles, dimensions]
        vel_trajectory = positions[1:,:,:] - positions[:-1,:,:]
        acc_trajectory = vel_trajectory[1:,:,:]- vel_trajectory[:-1,:,:]
    
        vels = np.concatenate((vels, vel_trajectory.reshape(-1, 2)), axis=0)
        accs = np.concatenate((accs, acc_trajectory.reshape(-1, 2)), axis=0)
    
    vel_mean = list(vels.mean(axis=0))
    vel_std = list(vels.std(axis=0))
    acc_mean = list(accs.mean(axis=0))
    acc_std = list(accs.std(axis=0))

    train_filepath = os.path.join(out_dir, 'train.npz')
    valid_filepath = os.path.join(out_dir, 'valid.npz')
    test_filepath = os.path.join(out_dir, 'test.npz')

    np.savez(train_filepath, trajectories=ds_train)
    np.savez(valid_filepath, trajectories=ds_valid)
    np.savez(test_filepath, trajectories=ds_test)
    
    print(f"{len(ds_train)} trajectories parsed and saved to train.npz.")
    print(f"{len(ds_valid)} trajectories parsed and saved to valid.npz.")
    print(f"{len(ds_test)} trajectories parsed and saved to test.npz.")
    
    # Save metadata
    out_file = os.path.join(out_dir, 'metadata.json')
    meta_data = {
        'sequence_length': positions.shape[0],
        'dt': 0.002 * step_size,
        'bounds': [[0, 1], [0, 1]],
        'default_connectivity_radius': 0.03,
        'vel_mean': vel_mean,
        'vel_std': vel_std,
        'acc_mean': acc_mean,
        'acc_std': acc_std,
        'file_train': train_info,
        'file_valid': valid_info,
        'file_test': test_info
    }
    
    with open(out_file, 'w') as f:
        json.dump(meta_data, f)
    
    print(meta_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process NPZ files and prepare datasets.')
    parser.add_argument('--dataset', type=str, default='Concrete2D-T-Step2', help='Dataset name')
    parser.add_argument('--step_size', type=int, default=2, help='Step size for downsampling')
    parser.add_argument('--total_step', type=int, default=100, help='Total steps to consider')
    parser.add_argument('--in_dir', type=str, required=True, help='Input directory containing npz files')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory to save processed data')
    
    args = parser.parse_args()
    process_data(args.dataset, args.step_size, args.total_step, args.in_dir, args.out_dir)
