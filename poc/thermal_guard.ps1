# Overnight heat guard. Checks the GPU temperature every 20 s.
#   >= 82 C : freeze all experiment processes inside WSL (SIGSTOP) so the PC cools down
#   <= 72 C : unfreeze them (SIGCONT); the run simply continues where it was
# Stops by itself when the overnight run finishes (or after 12 hours). Log: results\thermal.log
param([string]$RunLog = "C:\Folding\poc\results\overnight.log")
$PauseAt = 82; $ResumeAt = 72
$Log = "C:\Folding\poc\results\thermal.log"
$Names = "python phmmer jackhmmer mmseqs colabfold_batch USalign"
$paused = $false
$hot = 0
$end = (Get-Date).AddHours(12)
function Note($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content $Log }
function Signal($sig) { $cmd = (($Names -split " ") | ForEach-Object { "pkill -$sig -x $_" }) -join "; "; wsl -d Ubuntu -- bash -c "$cmd; true" | Out-Null }

Note "guard started (pause at $PauseAt C, resume at $ResumeAt C)"
while ((Get-Date) -lt $end) {
    $t = [int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    if ($t -gt $hot) { $hot = $t }
    if (-not $paused -and $t -ge $PauseAt) { Signal "STOP"; $paused = $true; Note "GPU $t C -> PAUSED all work" }
    elseif ($paused -and $t -le $ResumeAt) { Signal "CONT"; $paused = $false; Note "GPU $t C -> resumed" }
    if ((Get-Date).Minute % 30 -eq 0 -and (Get-Date).Second -lt 20) { Note "GPU $t C (max so far $hot C)" }
    if ((Test-Path $RunLog) -and (Select-String -Path $RunLog -Pattern "ALL DONE" -Quiet)) { break }
    Start-Sleep 20
}
if ($paused) { Signal "CONT" }
Note "guard finished (hottest GPU reading: $hot C)"
