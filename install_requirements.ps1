# path for venv
$venvPath = "venv"


if (!(Test-Path $venvPath)) {
    Write-Output "create virtual environment "
    python -m venv $venvPath
} else {
    Write-Output "virtual environment already existing"
}

Write-Output "Activating..."
& "$venvPath\Scripts\Activate.ps1"

$libraries = @(
    "torch_geometric",  "pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html",
    "git+https://github.com/pyt-team/TopoNetX.git",
    "git+https://github.com/pyt-team/TopoModelX.git",
    "networkx",
    "ogb"
)

foreach ($lib in $libraries) {
    Write-Output "Istalling $lib in venv..."
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    pip install $biblioteca
}

Write-Output "Sucessful!"
