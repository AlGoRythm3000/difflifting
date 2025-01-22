# Caminho do ambiente virtual
venvPath="venv"

# Cria o ambiente virtual, se ainda não existir
if [ ! -d "$venvPath" ]; then
    echo "Criando o ambiente virtual..."
    python3.10 -m venv $venvPath
else
    echo "Ambiente virtual já existe."
fi

# Ativa o ambiente virtual
echo "Ativando o ambiente virtual..."
source "$venvPath/bin/activate"

# Lista de bibliotecas a serem instaladas
bibliotecas=(
    "torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu"
    "torch_geometric"
    "pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0++cpu.html"
    "git+https://github.com/pyt-team/TopoNetX.git"
    "git+https://github.com/pyt-team/TopoModelX.git"
    "networkx"
    "ogb"
    "torchinfo==1.8.0"
)

# Instala as bibliotecas
for biblioteca in "${bibliotecas[@]}"; do
    echo "Instalando $biblioteca no ambiente virtual..."
    pip3 install $biblioteca
done

echo "Instalação completa!"
