<#
JudgeModel 판정 서버를 띄운다 (Windows · 127.0.0.1).

- 모델 : -Model 에 모델 이름을 주면 ~/.judgemodel/models/<모델> 을, 경로를 주면 그 폴더를 쓴다 (model.onnx 가 든 폴더).
- 기본은 --compat3 를 붙여 v1.0.0 과 같은 3칸 꼴로 답한다. `-NoCompat3` 를 주면 4칸 기본.
- 포트에 이미 무엇이 떠 있으면 새로 띄우지 않고 /health 만 찍고 끝낸다.
- 서버는 창 없는 python 프로세스로 남고, 로그는 ~/.judgemodel/logs/serve.{log,err} 에 쌓인다.

  .\serve.ps1 -Model <모델>             # 지금 띄운다
  .\serve.ps1 -Model <모델> -Register   # 이 계정으로 로그인할 때 자동으로 뜨게 작업 스케줄러에 올린다 (작업 이름 JudgeModel-NLI)
  .\serve.ps1 -Unregister               # 올린 작업을 내린다
#>
param(
    [string]$Model = "",         # 모델 이름(~/.judgemodel/models/<모델>) 또는 모델 폴더 경로
    [switch]$NoCompat3,          # 주면 --compat3 를 빼고 4칸 기본
    [int]$Port = 8092,
    [int]$Threads = 4,
    [string]$Python = "",
    [switch]$Register,
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"
$TaskName = "JudgeModel-NLI"
$ServePy = Join-Path (Split-Path $PSScriptRoot -Parent) "serve.py"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Output "작업 $TaskName 을 내렸다."
    return
}

if (-not $Model) { throw "-Model 을 준다 — 모델 이름(~/.judgemodel/models/<모델>) 또는 모델 폴더 경로." }
# 경로 구분자가 없으면 모델 이름으로 보고 ~/.judgemodel/models 아래에서 찾는다.
if ($Model -notmatch '[\\/]') { $Model = Join-Path $HOME ".judgemodel\models\$Model" }

if ($Register) {
    # 로그인 때 이 스크립트를 창 없이 한 번 돌린다. 스크립트는 서버를 띄우고 바로 끝난다.
    # 지금 받은 인자(모델 · 포트 · 칸 꼴 · 스레드 · python)를 작업에 그대로 실어 둔다.
    $arg = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$PSCommandPath`"" +
        " -Model `"$Model`" -Port $Port -Threads $Threads"
    # -File 로 넘기면 `$true 같은 값이 문자열이 되므로, 칸 꼴은 스위치 유무로만 싣는다.
    if ($NoCompat3) { $arg += " -NoCompat3" }
    if ($Python) { $arg += " -Python `"$Python`"" }
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
        -Description "JudgeModel NLI 판정 서버 (127.0.0.1:$Port)" -Force | Out-Null
    Write-Output "작업 $TaskName 을 올렸다 — 다음 로그인부터 자동으로 뜬다."
    return
}

function Get-Health {
    try { return Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3 } catch { return $null }
}

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    $h = Get-Health
    if ($h) { Write-Output "이미 떠 있다 : $($h | ConvertTo-Json -Compress)" }
    else { Write-Output "$Port 를 다른 프로그램이 쓰고 있다 (/health 응답 없음). 띄우지 않았다." }
    return
}

# python 고르기 : -Python 인자 → JudgeModel/venv-serve.
# onnxruntime · tokenizers · numpy 가 든 서빙 venv 다. 시스템 python 에는 numpy 가 없을 수 있다.
if (-not $Python) {
    $cand = Join-Path (Split-Path $PSScriptRoot -Parent) "venv-serve\Scripts\python.exe"
    if (Test-Path $cand) { $Python = $cand }
    else { throw "서빙 venv 가 없다. README 「띄우기」 대로 venv-serve 를 만들거나 -Python 을 준다." }
}
if (-not (Test-Path (Join-Path $Model "model.onnx"))) { throw "모델이 없다 : $Model" }

$logDir = Join-Path $HOME ".judgemodel\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir "serve.log"
$err = Join-Path $logDir "serve.err"

$pyArgs = @("-I", "`"$ServePy`"", "--model", "`"$Model`"", "--port", $Port, "--threads", $Threads)
if (-not $NoCompat3) { $pyArgs += "--compat3" }   # 4칸 모델을 v1.0.0 과 같은 3칸 꼴로
$p = Start-Process -FilePath $Python -WindowStyle Hidden -PassThru `
    -ArgumentList $pyArgs `
    -RedirectStandardOutput $log -RedirectStandardError $err

# 모델(약 2GB)을 올리는 데 몇 초 걸린다. /health 가 답할 때까지 최대 90초 기다린다.
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 1
    if ($p.HasExited) { throw "서버가 바로 죽었다 (종료 코드 $($p.ExitCode)). $err 를 본다." }
    $h = Get-Health
    if ($h) { Write-Output "띄웠다 (pid $($p.Id)) : $($h | ConvertTo-Json -Compress)"; return }
}
throw "90초 안에 /health 가 답하지 않았다. $log · $err 를 본다."
