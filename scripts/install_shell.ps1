# Makes "wake up buddy" work in BOTH PowerShell and Command Prompt.
#
# They share nothing: PowerShell functions live in a profile script, while
# cmd.exe only knows about real files on PATH. Installing one and not the other
# is how you get "'wake' is not recognized" in a Command Prompt while it works
# perfectly in PowerShell.
#
# Idempotent: re-running replaces the profile block between its markers and
# leaves PATH alone if the entry is already there.

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$entry = Join-Path $projectRoot "assistant.py"
$binDir = Join-Path $projectRoot "bin"

if (-not (Test-Path $python)) { throw "python not found at $python" }
if (-not (Test-Path $entry))  { throw "assistant.py not found at $entry" }
if (-not (Test-Path $binDir)) { throw "bin folder not found at $binDir" }

# ── PowerShell ───────────────────────────────────────────────────
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
    Write-Host "PowerShell: updated the existing block in $profilePath"
} else {
    $updated = ($existing.TrimEnd() + "`r`n`r`n" + $block + "`r`n")
    Write-Host "PowerShell: added to $profilePath"
}

# utf8 with BOM: Windows PowerShell 5.1 reads a BOM-less profile as ANSI and
# mangles any non-ASCII path in it.
Set-Content -Path $profilePath -Value $updated -Encoding UTF8

# ── Command Prompt ───────────────────────────────────────────────
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($null -eq $userPath) { $userPath = "" }
$entries = ($userPath -split ';') | Where-Object { $_ }

if ($entries -contains $binDir) {
    Write-Host "cmd.exe: $binDir is already on your PATH"
} else {
    $backup = Join-Path $env:USERPROFILE "user-path-backup.txt"
    $userPath | Out-File $backup -Encoding utf8

    # SetEnvironmentVariable rather than setx: setx silently truncates at 1024
    # characters, which quietly destroys a long PATH.
    [Environment]::SetEnvironmentVariable("Path", (($entries + $binDir) -join ';'), "User")
    Write-Host "cmd.exe: added $binDir to your PATH (previous value saved to $backup)"
}

Write-Host ""
Write-Host "Open a NEW terminal - PowerShell or Command Prompt - then:"
Write-Host "  wake up buddy"
