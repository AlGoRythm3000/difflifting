import torch
import numpy as np
import matplotlib.pyplot as plt
import os

import time 

FILE_PATH = "results/PROTEINS_diffLifting_GIN_HyperGAT_42_k_adaptative.results" 

def plot_results(file_path):
    if not os.path.exists(file_path):
        print(f"Erreur : Le fichier {file_path} n'existe pas.")
        return

    # loading resultats on the CPU from file 
    results = torch.load(file_path, map_location=torch.device('cpu'))
    
    # extrac and conver data to numpy for plotting
    train_losses = results["train_losses"].numpy()
    val_losses = results["val_losses"].numpy()
    test_losses = results["test_losses"].numpy()
    
    val_accuracies = results["val_accuracies"].numpy()
    test_accuracies = results["test_accuracies"].numpy()
    
    epochs = range(1, len(train_losses) + 1)

    # create graphs
    plt.figure(figsize=(14, 5))

    # loss graphs 
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o', markersize=3)
    plt.plot(epochs, test_losses, label='Test Loss', marker='o', markersize=3)
    plt.title('Évolution de la perte (Loss)')
    plt.xlabel('Époques')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # accuracy graphs
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', marker='o', markersize=3)
    plt.plot(epochs, test_accuracies, label='Test Accuracy', marker='o', markersize=3)
    plt.title('Évolution de la précision (Accuracy)')
    plt.xlabel('Époques')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle(file_path.split("/")[-1].replace(".results", ""))
    
    # display model parameters
    print("--- Paramètres du modèle ---")
    for key, value in results["params"].items():
        print(f"{key}: {value}")

    plt.tight_layout()
    plt.show()


def extract_final_score(file_path):
    if not os.path.exists(file_path):
        return None

    results = torch.load(file_path, map_location=torch.device('cpu'))
    
    val_accuracies = results["val_accuracies"].numpy()
    test_accuracies = results["test_accuracies"].numpy()
    
    # finding the best epoch based on validation accuracy
    best_epoch_idx = np.argmax(val_accuracies)
    
    # extracting the corresponding validation accuracy and test accuracy at that epoch
    best_val_score = val_accuracies[best_epoch_idx]
    final_test_score = test_accuracies[best_epoch_idx]
    
    print(f"Best epoch : {best_epoch_idx + 1}")
    print(f"Validation Accuracy at this epoch : {best_val_score:.4f}")
    print(f"Test Accuracy final to report : {final_test_score:.4f}")
    
    return final_test_score

def extract_score_silent(file_path):
    """
    Same as extract_final_score but without print statements, to be used in aggregate_results()
    """
    if not os.path.exists(file_path):
        return None

    results = torch.load(file_path, map_location=torch.device('cpu'))
    val_accuracies = results["val_accuracies"].numpy()
    test_accuracies = results["test_accuracies"].numpy()
    
    train_times = results.get("train_times", [])
    test_times = results.get("test_times", [])
    
    print(train_times)
    print(test_times)
    
    avg_train_time = np.mean(train_times) if len(train_times) > 0 else 0.0
    avg_test_time = np.mean(test_times) if len(test_times) > 0 else 0.0
    
    best_epoch_idx = np.argmax(val_accuracies)
    score = test_accuracies[best_epoch_idx]
    
    return score, avg_train_time, avg_test_time

def aggregate_results():
    import glob
    
    # dictionary to group results by experiment name
    # key: name of the experiment
    # value: dict with lists of scores, train_times, and test_times
    experiments = {}

    dossier_script = os.path.dirname(os.path.abspath(__file__))
    fichiers = glob.glob(os.path.join(dossier_script, "*.results"))

    for file_path in fichiers:
        filename = os.path.basename(file_path)
        
        parts = filename.split('_')
        
        if len(parts) >= 3 and parts[-3].isdigit():
            base_exp_name = "_".join(parts[:-3])
            
            # data extraction 
            res = extract_score_silent(file_path)
            
            if res is not None:
                score, t_train, t_test = res
                
                if base_exp_name not in experiments:
                    experiments[base_exp_name] = {"scores": [], "train_times": [], "test_times": []}
                
                experiments[base_exp_name]["scores"].append(score)
                experiments[base_exp_name]["train_times"].append(t_train)
                experiments[base_exp_name]["test_times"].append(t_test)

    # printing the aggregated results in a nice table format
    print("\n" + "="*125)
    print(f"{'Name of the experiment':<55} | {'Mean ± Std (Acc)':<17} | {'Train Time/ep':<13} | {'Test Time/ep':<13} | {'Seeds'}")
    print("="*125)

    for exp_name in sorted(experiments.keys()):
        scores = experiments[exp_name]["scores"]
        train_times = experiments[exp_name]["train_times"]
        test_times = experiments[exp_name]["test_times"]
        
        # moeans and stds
        mean_score = np.mean(scores) * 100
        std_score = np.std(scores) * 100
        mean_t_train = np.mean(train_times)
        mean_t_test = np.mean(test_times)
        num_seeds = len(scores)
        
        print(f"{exp_name:<55} | {mean_score:>6.2f} ± {std_score:>4.2f}   | {mean_t_train:>10.4f} s  | {mean_t_test:>10.4f} s  | {num_seeds}/5")
        
    print("="*125)


if __name__ == "__main__":
    # plot_results(FILE_PATH)
    # extract_final_score(FILE_PATH)
    # import os, glob
    
    import glob

    # glob.glob directly finds all files matching the pattern, we look for all .results files in the current directory
    # for filename in glob.glob("*.results"):
    #     print(f"\n--- {filename} ---")
    #     extract_final_score(filename)
    #     plot_results(filename)

    aggregate_results()