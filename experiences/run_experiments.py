import subprocess
import os

# Les datasets à tester pour la classification de graphes

# Available command-line arguments:
    # --seed (int): Random seed for reproducibility. Default is 42.
    # --gnn (str): Type of GNN to use, options are 'gcn', 'gin', 'linear'. Default is 'gcn'.
    # --tnn (str): Type of TNN to use, options are 'san', 'scn', 'sccn'. Default is 'san'.
    # --dataset (str): Dataset to use for the task, options include 'ogbg-molhiv',
        # 'ogbg-molpcba', 'NCI1', 'NCI109', 'IMDB-BINARY', 'ENZYMES', 'CORA', 
        # 'CITESEER', 'PUBMED'. Default is 'ogbg-molhiv'.
    # --lr (float): Learning rate for training. Default is 0.001.
    # --batch_size (int): Batch size for training. Default is 32.
    # --max_epochs (int): Maximum number of epochs to train the model. Default is 1000.
    # --early_stop_patience (int): Patience for early stopping. Default is 40.
    # --lr_decay_patience (int): Patience for learning rate decay. Default is 10.
    # --hidden_dim (int): Dimension of hidden layers. Default is 64.
    # --depth (int): Depth of the network. Default is 2.
    # --no-bn: Disable batch normalization if this flag is present.
    # --deepset_aggr_type (str): Aggregation type for DeepSet, options are 'sum', 'cat', 'mean'. Default is 'sum'.
    # --global_pooling (str): Global pooling method, options are 'sum', 'mean'. Default is 'mean'.


DATASETS = ["Cora", "Citeseer", "Pubmed", "CS", "Physics", "Cornell", 
            "Texas", "Wisconsin", "chameleon", "crocodile", "squirrel", 
            "ogbg-molhiv", "NCI1", "NCI109", "IMDB-BINARY", "REDDIT-BINARY", 
            "ENZYMES", "PROTEINS", "DD", "MUTAG", "ZINC"]

DATASETS_samples = ["NCI1", "NCI109", "MUTAG", "PROTEINS"]


# === GNN and TNN models ===
GNN_MODELS = ["GIN", "GPS"]  # GIN pour les petits datasets, GPS pour ZINC

TNN_MODELS = ["CWN","SCN2","CXN","UniGCNII","UniGIN","UniGCN","HyperGAT","TOPOTUNE"] 
TNN_MODELS_cellular = ["CWN", "CXN", "CIN"]
TNN_MODELS_hypergraph = ["UniGCN", "UniGIN"]
# ===================

# === lifitng methods ===
LIFTING_METHODS = ["diffLifting", 
                   "SimplicialCliqueLifting",
                   "SimplicialKHopLifting",
                   "CellCycleLifting",
                   "DiscreteConfigurationComplexLifting",
                   "diffLifting",
                   "HypergraphKHopLifting",
                   "HypergraphKNNLifting",
                   "HypergraphKernelLifting"]

LIFTING_METHODS_hypergraph = ["HypergraphKHopLifting", "HypergraphKNNLifting", "HypergraphKernelLifting", "diffLifting"]
LIFTING_METHODS_cellular = ["CellCycleLifting", "diffLifting"]
# ======================

# Graines aléatoires pour moyenner les résultats (très important en recherche)
SEEDS = [42, 9, 3] 

# Hyperparamètres par défaut
MAX_EPOCHS = 30
BATCH_SIZE = 32
LR = 0.001

def run_command(command):
    """Exécute une commande dans le terminal et gère les erreurs."""
    print(f"\n{'='*80}\nEXÉCUTION : {command}\n{'='*80}")
    try:
        # Lancement du processus
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de la commande : {e}")



if __name__ == "__main__":
    
    tests_courts = True  # Mettre à False pour lancer les expériences complètes (longues)
    tests_complets = False # Mettre à True pour lancer les expériences complètes (longues)
    
    os.makedirs("results", exist_ok=True)
    
    # courts tests sur qq datasets et sur les hypergraphes et les CCs pour vérifier que tout fonctionne avant de lancer les expériences complètes: 
    if tests_courts:
        # --- test accuracy sur les complexes cellulaires ---
        print("\n" + "#"*50)
        print("DÉMARRAGE DES EXPÉRIENCES : CCs")
        print("#"*50)

        for dataset in DATASETS_samples:
            for tnn in TNN_MODELS_cellular:
                for seed in SEEDS:
                    for lifting in LIFTING_METHODS_cellular:
                    # Pour les TUDatasets, GIN est le standard
                        cmd = (
                            f"python main_graph_classification.py "
                            f"--dataset {dataset} "
                            f"--gnn GIN "
                            f"--tnn {tnn} "
                            f"--lifting {lifting} "
                            f"--max_epochs {MAX_EPOCHS} "
                            f"--batch_size {BATCH_SIZE} "
                            f"--lr {LR} "
                            f"--seed {seed}"
                        )
                        run_command(cmd)


        # --- Test accuracy sur les hypergraphes ---
        print("\n" + "#"*50)
        print("DÉMARRAGE DES EXPÉRIENCES : hypergraph")
        print("#"*50)

        for dataset in DATASETS_samples:
            for tnn in TNN_MODELS_hypergraph:
                for seed in SEEDS:
                    for lifting in LIFTING_METHODS_hypergraph:
                        cmd = (
                            f"python main_graph_classification.py "
                            f"--dataset {dataset} "
                            f"--gnn GPS "
                            f"--tnn {tnn} "
                            f"--lifting {lifting} "
                        f"--max_epochs {MAX_EPOCHS} " 
                        f"--batch_size {BATCH_SIZE} "          # Batch size plus grand pour les grands datasets
                        f"--lr {LR} "              
                        f"--seed {seed}"
                    )
                    run_command(cmd)

        print("\nexpériences terminées ")
        
    if tests_complets:
        pass