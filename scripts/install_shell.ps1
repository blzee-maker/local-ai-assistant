# Adds "wake up buddy" (and friends) to your PowerShell profile.
#
# Idempotent: re-running replaces the block between the markers rather than
# appending a second copy. Remove the block by hand to uninstall.

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$entry = Join-Path $projectRoot "assistant.py"

if (-not (Test-Path $python)) { throw "python not found at $python" }
if (-not (Test-Path $entry))  { throw "assistant.py not found at $entry" }

$begin = "# >>> local-ai-assistant >>>"
$end   = "# <<< local-ai-assistant <<<"

$block = @"
$begin
# One command to bring the assistant up and start talking.
`$env:LOCAL_AI_HOME = "$projectRoot"

function Invoke-Assistant { & "$python" "$entry" @args }
Set-Alias ai Invoke-Assistant

# "wake", "wake up", "wake up buddy" all work: extra words are ignored.
function wake  { Invoke-Assistant wake @args }
function buddy { Invoke-Assistant wake @args }
$end
"@

$profilePath = $PROFILE.CurrentUserAllHosts
$profileDir = Split-Path -Parent $profilePath
if (-not (Test-Path $profileDir)) { New-Item -ItemType Directory -Path $profileDir -Force | Out-Null }
if (-not (Test-Path $profilePath)) { New-Item -ItemType File -Path $profilePath | Out-Null }

$existing = Get-Content $profilePath -Raw -ErrorAction SilentlyContinue
if ($null -eq $existing) { $existing = "" }

if ($existing -match [regex]::Escape($begin)) {
    $pattern = "(?s)" + [regex]::Escape($begin) + ".*?" + [regex]::Escape($end)
    $updated = [regex]::Replace($existing, $pattern, [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $block })
    Write-Host "Updated the existing block in $profilePath"
} else {
    $updated = ($existing.TrimEnd() + "`r`n`r`n" + $block + "`r`n")
    Write-Host "Added to $profilePath"
}

# utf8 with BOM: Windows PowerShell 5.1 reads a BOM-less profile as ANSI and
# mangles any non-ASCII path in it.
Set-Content -Path $profilePath -Value $updated -Encoding UTF8

Write-Host ""
Write-Host "Open a NEW terminal, then:"
Write-Host "  wake up buddy"
