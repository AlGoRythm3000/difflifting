
venvPath="myvenv"


if [ ! -d "$venvPath" ]; then
    echo "Create virtual environment..."
    python3.10 -m venv $venvPath
else
    echo "Virtual Environment already exists."
fi

echo "Activating venv"
source "$venvPath/bin/activate"

libraries=(
    "torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu"
    "torch_geometric"
    "torchinfo"
    "pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html"
    "git+https://github.com/pyt-team/TopoNetX.git"
    "git+https://github.com/pyt-team/TopoModelX.git"
    "networkx"
    "ogb"
    "torchinfo==1.8.0"
)

for biblioteca in "${bibliotecas[@]}"; do
    echo "Installing $biblioteca in venv..."
    pip3 install $biblioteca
done

echo "Installation sucessful!"
