import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Remplace par le chemin exact de ton fichier généré dans le dossier "results/"
# Exemple de nom : "PROTEINS_diffLifting_GIN_UniGIN_42_k_adaptative.results"
FILE_PATH = "results/PROTEINS_diffLifting_GIN_HyperGAT_42_k_adaptative.results" 

def plot_results(file_path):
    if not os.path.exists(file_path):
        print(f"Erreur : Le fichier {file_path} n'existe pas.")
        return

    # Charger les résultats
    # Les tenseurs étant sur CPU ou GPU, on s'assure de les charger sur le CPU pour l'affichage
    results = torch.load(file_path, map_location=torch.device('cpu'))
    
    # Extraire les données et les convertir en listes numpy
    train_losses = results["train_losses"].numpy()
    val_losses = results["val_losses"].numpy()
    test_losses = results["test_losses"].numpy()
    
    val_accuracies = results["val_accuracies"].numpy()
    test_accuracies = results["test_accuracies"].numpy()
    
    epochs = range(1, len(train_losses) + 1)

    # Créer les graphiques
    plt.figure(figsize=(14, 5))

    # Graphique des pertes (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o', markersize=3)
    plt.plot(epochs, test_losses, label='Test Loss', marker='o', markersize=3)
    plt.title('Évolution de la perte (Loss)')
    plt.xlabel('Époques')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Graphique des précisions (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', marker='o', markersize=3)
    plt.plot(epochs, test_accuracies, label='Test Accuracy', marker='o', markersize=3)
    plt.title('Évolution de la précision (Accuracy)')
    plt.xlabel('Époques')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle(FILE_PATH.split("/")[-1].replace(".results", ""))
    
    # Afficher les paramètres utilisés pour ce run
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
    
    # Trouver l'époque avec la meilleure validation (attention, argmax donne l'index, donc époque - 1)
    best_epoch_idx = np.argmax(val_accuracies)
    
    # Extraire le score de test à CETTE époque précise
    best_val_score = val_accuracies[best_epoch_idx]
    final_test_score = test_accuracies[best_epoch_idx]
    
    print(f"Meilleure époque : {best_epoch_idx + 1}")
    print(f"Validation Accuracy à cette époque : {best_val_score:.4f}")
    print(f"Test Accuracy finale à rapporter : {final_test_score:.4f}")
    
    return final_test_score


if __name__ == "__main__":
    plot_results(FILE_PATH)
    extract_final_score(FILE_PATH)
    