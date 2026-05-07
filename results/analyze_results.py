import torch
import numpy as np
import matplotlib.pyplot as plt
import os

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
    
    best_epoch_idx = np.argmax(val_accuracies)
    return test_accuracies[best_epoch_idx]

def aggregate_results():
    import glob
    
    # dictionnary to groupe results by experiment name (without seed and suffix)
    # key : name of the experiment (ex: MUTAG_diffLifting_GPS_UniGCN)
    # value : list of test accuracies for different seeds (ex: [0.85, 0.87, 0.86])
    experiments = {}

    dossier_script = os.path.dirname(os.path.abspath(__file__))
    fichiers = glob.glob(os.path.join(dossier_script, "*.results"))

    for file_path in fichiers:
        filename = os.path.basename(file_path)
        
        # isolating the seed and the suffix to get the base name of the experiment
        parts = filename.split('_')
        
        if len(parts) >= 3 and parts[-3].isdigit():
            base_exp_name = "_".join(parts[:-3])
            
            # extracting the final test score for this experiment
            score = extract_score_silent(file_path)
            
            if score is not None:
                if base_exp_name not in experiments:
                    experiments[base_exp_name] = []
                experiments[base_exp_name].append(score)

    # displaying the results in a formatted table
    print("\n" + "="*90)
    print(f"{'Name of the experiment':<55} | {'Mean ± Std':<20} | {'Seeds'}")
    print("="*90)

    # tri par ordre alphabetique
    for exp_name in sorted(experiments.keys()):
        scores = experiments[exp_name]
        
        # converting in percentage and calculating mean and std
        mean_score = np.mean(scores)*100
        std_score = np.std(scores)*100
        num_seeds = len(scores)
        
        print(f"{exp_name:<55} | {mean_score:>6.2f} ± {std_score:>4.2f}   | {num_seeds}/5")
        
    print("="*90)


if __name__ == "__main__":
    # plot_results(FILE_PATH)
    # extract_final_score(FILE_PATH)
    # import os, glob
    
    import glob

    # glob.glob directly finds all files matching the pattern, we look for all .results files in the current directory
    for filename in glob.glob("*.results"):
        print(f"\n--- {filename} ---")
        extract_final_score(filename)
        # plot_results(filename)

    aggregate_results()