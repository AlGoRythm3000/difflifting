import subprocess
import os

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


def run_one_experiment(dataset, tnn, lifting, list_seed:list):
    
    for seed in list_seed:
        """Lance une expérience unique avec les paramètres spécifiés."""
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

if __name__ == "__main__":
    
    os.makedirs("results", exist_ok=True)
    
    # target datasets for the experiments
    DATASETS_samples = [ #"NCI1", 
                        # "NCI109", 
                        # "MUTAG", 
                        "PROTEINS"]

    # random seeds for reproducibility
    SEEDS = [42, 9, 3, 13, 28] 

    # default hyperparameters 
    MAX_EPOCHS = 200
    BATCH_SIZE = 32
    LR = 0.005
    
    # ==========================================
    # cellular domain
    # ==========================================
    # TNN : CWN, CXN
    # Lifting : Cycle (CellCycleLifting), difflift (diffLifting)
    
    all_experiments = False
    one_experiment = True
    
    if all_experiments == True:
        tnns_cellular = ["CWN", "CXN"]
        liftings_cellular = ["CellCycleLifting", "diffLifting"]
        
        print("\n" + "#"*50)
        print("DÉMARRAGE DES EXPÉRIENCES : CELLULAR")
        print("#"*50)

        
        for tnn in tnns_cellular:
            for lifting in liftings_cellular:
                for seed in SEEDS:
                    cmd = (
                        f"python main_graph_classification.py "
                        f"--dataset PROTEINS "
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

        # for dataset in DATASETS_samples:
        for tnn in tnns_hypergraph:
            for lifting in liftings_hypergraph:
                for seed in SEEDS:
                    cmd = (
                        f"python main_graph_classification.py "
                        f"--dataset PROTEINS "
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
    
    if one_experiment == True:
        # running 1 specific experiment that maybe didnt work well the first time, to check if it works and to analyze results with analyze_results.py
        
        experiment_parameters = {
            "dataset": "PROTEINS",
            "tnn": "CXN",
            "lifting": "CellCycleLifting",
            "list_seed": SEEDS
        }
        
        run_one_experiment(
            **experiment_parameters
        )