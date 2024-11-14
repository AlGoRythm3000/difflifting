# Caminho do ambiente virtual
$venvPath = "venv"

# Cria o ambiente virtual, se ainda não existir
if (!(Test-Path $venvPath)) {
    Write-Output "Criando o ambiente virtual..."
    python -m venv $venvPath
} else {
    Write-Output "Ambiente virtual já existe."
}

# Ativa o ambiente virtual
Write-Output "Ativando o ambiente virtual..."
& "$venvPath\Scripts\Activate.ps1"

# Lista de bibliotecas a serem instaladas
$bibliotecas = @(
    "torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121",
    "torch_geometric",  "pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html",
    "git+https://github.com/pyt-team/TopoNetX.git",
    "git+https://github.com/pyt-team/TopoModelX.git",
    "networkx",
    "ogb"
)

# Instala as bibliotecas
foreach ($biblioteca in $bibliotecas) {
    Write-Output "Instalando $biblioteca no ambiente virtual..."
    pip install $biblioteca
}

Write-Output "Instalação completa!"
