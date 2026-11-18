# Download the 20-shard list to F:\atlas_shards one at a time (Windows curl, sequential).
# Writes <name>.ok after each finished file so the WSL converter knows it is complete.
param([string]$List = "C:\Folding\poc\results\atlas20_shards.txt")
$Dir = "F:\atlas_shards"
$Log = "C:\Folding\poc\results\atlas20_download.log"
New-Item -ItemType Directory -Force $Dir | Out-Null
foreach ($u in Get-Content $List) {
    if (-not $u) { continue }
    $n = Split-Path $u -Leaf
    if ($n -like "tm_.80_.90_plddt_.80_.90_00.npz") { continue }           # already inside WSL
    if (Test-Path "$Dir\$n.ok") { continue }
    for ($k = 0; $k -lt 4; $k++) {
        $r = curl.exe -sS --retry 5 -C - -o "$Dir\$n" -w "%{size_download} %{time_total} %{speed_download}" $u
        if ($LASTEXITCODE -eq 0) { "" | Out-File "$Dir\$n.ok"; break }
        Start-Sleep 10
    }
    "$(Get-Date -Format 'HH:mm:ss') $n bytes/seconds/speed: $r exit $LASTEXITCODE" | Add-Content $Log
}
"$(Get-Date -Format 'HH:mm:ss') DOWNLOADS DONE" | Add-Content $Log
