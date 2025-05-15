import subprocess
import os
from colorama import Fore, Style, init
import numpy as np
import mkl
init(autoreset=True)

def check_dataset_args(dataset):
    if dataset == "Cora":
        dataset_args = ["--lr", str(0.001),
                        "--hidden_dim", str(64),
                        "--num_layers", str(2),
                        "--dataset", dataset,
                        ]
    elif dataset == "Citeseer":
        dataset_args = ["--lr", str(0.001),
                        "--hidden_dim", str(128),
                        "--num_layers", str(2),
                        "--dataset", dataset,
                        ]
    elif dataset == "Pubmed":
        dataset_args = ["--lr", str(0.001),
                        "--hidden_dim", str(128),
                        "--num_layers", str(2),
                        "--dataset", dataset,
                        ]
#    elif dataset == "Texas" or dataset=="Wisconsin":
#        dataset_args = ["--lr", str(0.001),
#                        "--hidden_dim", str(64),
#                        "--num_layers", str(2),
#                        "--dataset", dataset,
#                       "--weight_decay",str(5e-6)
#                    ]
    elif dataset == "Texas" :
        dataset_args = ["--lr", str(0.005),
                        "--weight_decay", str(5e-6),
                        "--batch_size", str(1),
                        "--hidden_dim", str(64),
                        "--num_layers", str(1),
                        "--dataset", dataset,

               ]
    elif dataset=="Wisconsin":
        dataset_args = ["--lr", str(0.01),
                        "--weight_decay", str(5e-6),
                        "--batch_size", str(1),
                        "--hidden_dim", str(64),
                        "--num_layers", str(1),
                        "--dataset", dataset,

               ]
    elif dataset == "CS" or dataset== "Physics":
        dataset_args = ["--lr", str(0.001),
                        "--hidden_dim", str(128),
                        "--num_layers", str(2),
                        "--dataset", dataset,
                        ]
    elif dataset == "chameleon" or dataset=="squirrel":
        dataset_args = ["--lr", str(0.001),
                        "--hidden_dim", str(64),
                        "--num_layers", str(2),
                        "--dataset", dataset,
                        ]
    return dataset_args
# Lista dos valores de tnn e seed a serem testados
tnn_options = ["UniGCNII", "UniGIN"]
datasets = ["Cora", "Citeseer","Texas","Wisconsin"]
# datasets = [ "NCI1", "NCI109", "MUTAG", "PROTEINS", "ZINC"]
gnn = "gin"
embedding_dims=[32,64, 128]
num_layers_gnn=[2, 3]
k_max_values=[3, 5, 7, 9, 11]
# t_values = [round(x, 2) for x in np.arange(0.1,i10.01, 0.5)]
temps=[0.1, 0.6, 1.1, 1.6, 2.1, 2.6, 3.1, 3.6, 4.1, 4.6, 5.1, 5.6, 6.1, 6.6, 7.1, 7.6, 8.1, 8.6, 9.1, 9.6]
seed_options = [42, 3, 9]
# Caminho para o script principal
script_path = "main.py"

# Garante que a pasta de logs exista
os.makedirs("logs", exist_ok=True)
venv_python = "python3"

for dataset in datasets:

    for tnn in tnn_options:
        for embedding_dim in embedding_dims:
            for temp in temps:
         #       for k_max in k_max_values:
                    for seed in seed_options:
                        # for t in t_values:
                            logdir = f"experimentos_kernel_hypergraph/kernel_hypergraph_{tnn}_adaptk_v_kmax/{dataset}_{gnn}_{embedding_dim}_{seed}_{temp}"
                            log_path = f"logs/{tnn}_kernel_hypergraph_seed{seed}.txt"

                            print(Fore.CYAN + f"\n[RUNNING] --tnn {tnn}, --seed {seed}\n{'-' * 50}")

                            other_configs = [
                                            "--t",str(temp),
                                            "--lifting", "HypergraphKernelLifting",
                                            "--logdir", logdir,
                                            "--tnn", tnn,
                                            "--seed", str(seed)]
                            cmd = [
                                venv_python,
                                script_path,


                            ] + check_dataset_args(dataset) + other_configs
                            with open(log_path, "w",  encoding="utf-8") as log_file:
                                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,

                                                           encoding='utf-8',
                                                           errors='replace'
                                                           )

                                for line in process.stdout:
                                    print(Fore.WHITE + line, end="")  # imprime colorido
                                    log_file.write(line)

                                process.wait()

                                if process.returncode == 0:
                                    print(Fore.GREEN + f"\n[SUCCESS] Finalizado com sucesso para --tnn {tnn}, --seed {seed}")
                                else:
                                    print(Fore.RED + f"\n[ERROR] Código de retorno {process.returncode} para --tnn {tnn}, --seed {seed}")
                                    print(Fore.YELLOW + f"[LOG] Veja detalhes em: {log_path}")