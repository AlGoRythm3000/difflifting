import subprocess
import os

# target datasets for the experiments
DATASETS_samples = ["NCI1", "NCI109", "MUTAG", "PROTEINS"]

# random seeds for reproducibility
SEEDS = [42, 9, 3] 

# default hyperparameters 
MAX_EPOCHS = 20 
BATCH_SIZE = 32
LR = 0.001

def run_command(command):
    """Exécute une commande dans le terminal et gère les erreurs."""
    print(f"\n{'='*80}\nEXÉCUTION : {command}\n{'='*80}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erreur ignorée pour continuer la suite : {e}")
    except KeyboardInterrupt:
        print("\nInterruption manuelle par l'utilisateur.")
        exit()


if __name__ == "__main__":
    
    os.makedirs("results", exist_ok=True)
    
    # ==========================================
    # cellular domain
    # ==========================================
    # TNN : CWN, CXN
    # Lifting : Cycle (CellCycleLifting), ∂lift (diffLifting)
    
    tnns_cellular = ["CWN", "CXN"]
    liftings_cellular = ["CellCycleLifting", "diffLifting"]
    
    print("\n" + "#"*50)
    print("DÉMARRAGE DES EXPÉRIENCES : CELLULAR")
    print("#"*50)

    for dataset in DATASETS_samples:
        for tnn in tnns_cellular:
            for lifting in liftings_cellular:
                for seed in SEEDS:
                    cmd = (
                        f"python main_graph_classification.py "
                        f"--dataset {dataset} "
                        # f"--gnn GIN " 
                        f"--tnn {tnn} "
                        f"--lifting {lifting} "
                        f"--max_epochs {MAX_EPOCHS} "
                        f"--batch_size {BATCH_SIZE} "
                        f"--lr {LR} "
                        f"--seed {seed}"
                    )
                    run_command(cmd)


    # ==========================================
    # hypergraph domain 
    # ==========================================
    # TNN : UniGCN2 (UniGCN), UniGIN
    # Lifting : k-hop, k-NN, kernel, ∂lift (diffLifting)
    
    tnns_hypergraph = ["UniGCN", "UniGIN"]
    liftings_hypergraph = [
        "HypergraphKHopLifting", 
        "HypergraphKNNLifting", 
        "HypergraphKernelLifting", 
        "diffLifting"
    ]

    print("\n" + "#"*50)
    print("DÉMARRAGE DES EXPÉRIENCES : HYPERGRAPH")
    print("#"*50)

    for dataset in DATASETS_samples:
        for tnn in tnns_hypergraph:
            for lifting in liftings_hypergraph:
                for seed in SEEDS:
                    cmd = (
                        f"python main_graph_classification.py "
                        f"--dataset {dataset} "
                        # f"--gnn GIN " 
                        f"--tnn {tnn} "
                        f"--lifting {lifting} "
                        f"--max_epochs {MAX_EPOCHS} " 
                        f"--batch_size {BATCH_SIZE} " 
                        f"--lr {LR} "              
                        f"--seed {seed}"
                    )
                    run_command(cmd)

    print("\nToutes les expériences du Tableau 1 ont été lancées.")