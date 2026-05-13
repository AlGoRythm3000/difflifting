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


def run_one_experiment(dataset, gnn, tnn, lifting, list_seed:list):
    
    for seed in list_seed:
        """Lance une expérience unique avec les paramètres spécifiés."""
        cmd = (
            f"python main_graph_classification.py "
            f"--dataset {dataset} "
            f"--gnn {gnn} "
            f"--tnn {tnn} "
            f"--lifting {lifting} "
            f"--max_epochs {MAX_EPOCHS} "
            f"--batch_size {BATCH_SIZE} "
            f"--lr {LR} "
            f"--deepset_aggr_type mean "
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
    BATCH_SIZE = 256
    LR = 0.005
    
    all_experiments = False
    one_experiment = True  # Set to True to run a specific experiment for analysis with analyze_results.py
    
    if all_experiments == True:
        
        # ==========================================
        # cellular domain
        # ==========================================
        # TNN : CWN, CXN
        # Lifting : Cycle (CellCycleLifting), difflift (diffLifting)
        
        tnns_cellular = ["CWN", "CXN"]
        liftings_cellular = [
            # "CellCycleLifting", 
            "diffLifting"]
        
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
                        f"--deepset_aggr_type mean "
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
            # "HypergraphKNNLifting", 
            # "HypergraphKernelLifting", 
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
                        f"--deepset_aggr_type mean "              
                        f"--seed {seed}"
                    )
                    run_command(cmd)

        print("\nToutes les expériences du Tableau 1 ont été lancées.")
    
    if one_experiment == True:
        # running 1 specific experiment that maybe didnt work well the first time, to check if it works and to analyze results with analyze_results.py
        
        experiment_parameters = {
            "dataset": "PROTEINS",
            "gnn": "GIN",
            "tnn": "UniGCN",
            "lifting": "diffLifting",
            "list_seed": SEEDS
        }
        
        # usage: main_graph_classification.py [-h] [--seed SEED] [--gnn {GIN,GPS}] [--tnn {CWN,SCN2,CXN,UniGCNII,UniGIN,UniGCN,HyperGAT,TOPOTUNE}]
        #                             [--dataset {Cora,Citeseer,Pubmed,CS,Physics,Cornell,Texas,Wisconsin,chameleon,crocodile,squirrel,ogbg-molhiv,NCI1,NCI109,IMDB-BINARY,REDDIT-BINARY,ENZYMES,PROTEINS,DD,MUTAG,ZINC}]
        #                             [--lifting {SimplicialCliqueLifting,SimplicialKHopLifting,CellCycleLifting,DiscreteConfigurationComplexLifting,diffLifting,HypergraphKHopLifting,HypergraphKNNLifting,HypergraphKernelLifting}]
        #                             [--lr LR] [--weight_decay WEIGHT_DECAY] [--batch_size BATCH_SIZE] [--num_layers NUM_LAYERS]
        #                             [--num_layers_gnn NUM_LAYERS_GNN] [--max_epochs MAX_EPOCHS] [--number_of_mask NUMBER_OF_MASK]
        #                             [--early_stop_patience EARLY_STOP_PATIENCE] [--lr_decay_patience LR_DECAY_PATIENCE] [--logdir LOGDIR]
        #                             [--hidden_dim HIDDEN_DIM] [--gnn_embedding_dim GNN_EMBEDDING_DIM] [--k_max K_MAX] [--k K]
        #                             [--graph_transformer_n_heads GRAPH_TRANSFORMER_N_HEADS] [--positional_encoder_dim POSITIONAL_ENCODER_DIM]
        #                             [--positional_walking_len POSITIONAL_WALKING_LEN] [--depth DEPTH] [--no_readout] [--signed SIGNED]
        #                             [--use_dcm_split] [--no-bn] [--deepset_aggr_type {sum,cat,mean}] [--sub_gccn_model {GAT,GCN,GIN}]
        #                             [--global_pooling {sum,mean}] [--t T] [--deterministic]

        # DiffLifting for GNN tasks

        # options:
        # -h, --help            show this help message and exit
        # --seed SEED           Random seed.
        # --gnn {GIN,GPS}
        # --tnn {CWN,SCN2,CXN,UniGCNII,UniGIN,UniGCN,HyperGAT,TOPOTUNE}
        # --dataset {Cora,Citeseer,Pubmed,CS,Physics,Cornell,Texas,Wisconsin,chameleon,crocodile,squirrel,ogbg-molhiv,NCI1,NCI109,IMDB-BINARY,REDDIT-BINARY,ENZYMES,PROTEINS,DD,MUTAG,ZINC}
        # --lifting {SimplicialCliqueLifting,SimplicialKHopLifting,CellCycleLifting,DiscreteConfigurationComplexLifting,diffLifting,HypergraphKHopLifting,HypergraphKNNLifting,HypergraphKernelLifting}
        # --lr LR               Learning rate.
        # --weight_decay WEIGHT_DECAY
        #                         Weight Decay.
        # --batch_size BATCH_SIZE
        #                         Batch size.
        # --num_layers NUM_LAYERS
        #                         Number of tnn layers.
        # --num_layers_gnn NUM_LAYERS_GNN
        #                         Number of gnn layers
        # --max_epochs MAX_EPOCHS
        #                         Number of epochs to train.
        # --number_of_mask NUMBER_OF_MASK
        #                         if the dataset is heterophyllic you have to choose from 0 to 9
        # --early_stop_patience EARLY_STOP_PATIENCE
        # --lr_decay_patience LR_DECAY_PATIENCE
        # --logdir LOGDIR       Log directory
        # --hidden_dim HIDDEN_DIM
        # --gnn_embedding_dim GNN_EMBEDDING_DIM
        # --k_max K_MAX
        # --k K
        # --graph_transformer_n_heads GRAPH_TRANSFORMER_N_HEADS
        # --positional_encoder_dim POSITIONAL_ENCODER_DIM
        # --positional_walking_len POSITIONAL_WALKING_LEN
        # --depth DEPTH
        # --no_readout
        # --signed SIGNED
        # --use_dcm_split
        # --no-bn
        # --deepset_aggr_type {sum,cat,mean}
        # --sub_gccn_model {GAT,GCN,GIN}
        # --global_pooling {sum,mean}
        # --t T                 Temperature parameter for the heat kernel.
        # --deterministic       Run without sampling, use deterministic neighbor and inclusion selection.
                
        run_one_experiment(
            **experiment_parameters
        )
        
