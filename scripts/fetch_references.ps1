param(
    [string]$Destination = "external"
)

$ErrorActionPreference = "Stop"
$workspace = [System.IO.Path]::GetFullPath((Get-Location).Path)
$destinationPath = [System.IO.Path]::GetFullPath((Join-Path $workspace $Destination))

if (-not $destinationPath.StartsWith($workspace + [System.IO.Path]::DirectorySeparatorChar)) {
    throw "Destination must stay inside the current workspace."
}

New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null

$repositories = @(
    @{ Name = "chunbaobao-deepjscc"; Url = "https://github.com/chunbaobao/Deep-JSCC-PyTorch.git"; Commit = "2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06" },
    @{ Name = "deepjscc-feedback"; Url = "https://github.com/kurka/deepJSCC-feedback.git"; Commit = "61a853baf019d589a7be09dc443c72985b891a74" },
    @{ Name = "torchsig"; Url = "https://github.com/TorchDSP/torchsig.git"; Commit = "9b1949eef536df70b155ecceb6cf2b8de376e72b" },
    @{ Name = "rfml"; Url = "https://github.com/brysef/rfml.git"; Commit = "dbd3db2f6090232ebebc9d7107d990dd04fcb323" },
    @{ Name = "backdoorbench"; Url = "https://github.com/SCLBD/BackdoorBench.git"; Commit = "f02e3534645f0ee63d6848653062cd6c0d6c400d" }
)

foreach ($repository in $repositories) {
    $target = Join-Path $destinationPath $repository.Name
    if (Test-Path -LiteralPath $target) {
        Write-Warning "Skipping existing path: $target"
        continue
    }

    git clone --filter=blob:none --no-checkout $repository.Url $target
    git -C $target fetch --depth 1 origin $repository.Commit
    git -C $target checkout --detach $repository.Commit
}

Write-Output "Pinned research repositories are available under $destinationPath"
