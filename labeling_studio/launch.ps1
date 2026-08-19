<#
================================================================================
 YOLO Dataset Studio 런처
================================================================================
 .bat 은 실행 정책 우회용 껍데기이고 실제 로직은 여기에 있다.
 배치 파일에는 한글을 넣을 수 없다 — cmd.exe 가 UTF-8 한글을 파싱하다 줄
 동기화를 잃고 주석 조각을 명령으로 실행하려 든다. 두 번 겪었다.

 하는 일
   0. .venv 가 없으면 만들고 의존성을 설치한다 (첫 실행)
   1. venv 와 스크립트 존재 확인
   2. 이미 실행 중인지 확인 (같은 작업공간을 두 창에서 열면 라벨이 덮어써진다)
   3. 직전 실행이 오류로 끝났으면 알림
   4. pythonw 로 조용히 띄움 (-Console 이면 python 으로 콘솔 표시)
================================================================================
#>
[CmdletBinding()]
param(
    [switch]$Console,          # 콘솔에 출력을 표시하며 실행
    [switch]$Force,            # 중복 실행 확인 프롬프트를 건너뛴다 (자동화/테스트용)
    [switch]$Setup,            # venv 가 이미 있어도 설치를 다시 한다
    [switch]$SetupOnly,        # 설치만 하고 앱은 띄우지 않는다
    [switch]$Worker,           # 스튜디오 대신 학습 워커를 띄운다
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest            # 스튜디오에 그대로 넘길 인자 (데이터셋 경로 등)
)

$ErrorActionPreference = 'Stop'
$Root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$App    = Join-Path $Root 'yolo_dataset_studio.py'
$PyW    = Join-Path $Root '.venv\Scripts\pythonw.exe'
$Py     = Join-Path $Root '.venv\Scripts\python.exe'
$LogDir = Join-Path $Root 'docs'
$Log    = Join-Path $LogDir 'error.log'

function Fail([string]$msg) {
    Write-Host ""
    Write-Host "  [실행 불가] $msg" -ForegroundColor Red
    Write-Host ""
    # 콘솔 모드는 어차피 끝에서 멈춘다. -Force 는 자동 실행이라 멈추면 안 된다.
    if (-not $Console -and -not $Force) {
        Read-Host "  엔터를 누르면 닫힙니다" | Out-Null
    }
    exit 1
}

# ---- 0. 첫 실행이면 환경을 만든다 ------------------------------------------
#
#  torch 를 requirements.txt 에 넣지 않고 여기서 따로 까는 이유:
#  `pip install ultralytics` 는 torch 를 의존성으로 끌고 오는데, PyPI 기본 휠은
#  GPU 세대에 따라 못 돌 수 있다. Blackwell(RTX 50xx · sm_120)은 안정판 CUDA
#  빌드로 실행되지 않아 나이트리가 필요하다. 그래서 **torch 를 먼저** 맞는
#  인덱스에서 깔고, 그 다음 requirements 를 깐다 -- 이미 충족돼 있으므로
#  ultralytics 가 torch 를 다시 건드리지 않는다.

function Find-SystemPython {
    foreach ($c in @(@('py', '-3'), @('python'), @('python3'))) {
        $exe = $c[0]
        $pre = @()
        if ($c.Count -gt 1) { $pre = $c[1..($c.Count - 1)] }
        try {
            $v = & $exe @pre -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        } catch { continue }
        if ($LASTEXITCODE -eq 0 -and $v -match '^3\.(9|1\d)') {
            # 쉼표로 감싸지 않으면 PowerShell 이 배열을 풀어 버리고,
            # 감싸기 전에 더하지 않으면 배열이 중첩된다. 순서가 중요하다.
            $out = @($exe) + $pre
            return , $out
        }
    }
    return $null
}

function Get-TorchIndex {
    # GPU 이름으로 인덱스를 고른다. nvidia-smi 가 없으면 CPU 판.
    try {
        $name = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
    } catch { $name = $null }
    if (-not $name) {
        return @{ url = 'https://download.pytorch.org/whl/cpu'; pre = $false; note = 'CPU (NVIDIA GPU 를 못 찾음)' }
    }
    Write-Host "  GPU: $name" -ForegroundColor DarkGray
    if ($name -match 'RTX\s*50\d\d|B\d{3}0') {
        # Blackwell 은 안정판 CUDA 빌드로 실행되지 않는다
        return @{ url = 'https://download.pytorch.org/whl/nightly/cu130'; pre = $true; note = 'CUDA 13.0 나이트리 (Blackwell)' }
    }
    return @{ url = 'https://download.pytorch.org/whl/cu124'; pre = $false; note = 'CUDA 12.4 안정판' }
}

function Initialize-Venv {
    $req = Join-Path $Root 'requirements.txt'
    if ((Test-Path $Py) -and -not $Setup) {
        if ($SetupOnly) { Write-Host "  이미 설치되어 있습니다: $Py" -ForegroundColor DarkGray }
        return
    }

    Write-Host ""
    Write-Host "  ┌─ 첫 실행 준비 ─────────────────────────────────" -ForegroundColor Cyan
    Write-Host "  │  가상환경과 의존성을 설치합니다. 몇 분 걸립니다." -ForegroundColor Cyan
    Write-Host "  │  (torch 만 2GB 안팎입니다)" -ForegroundColor DarkGray
    Write-Host "  └────────────────────────────────────────────────" -ForegroundColor Cyan
    Write-Host ""

    if (-not (Test-Path $Py)) {
        $sys = Find-SystemPython
        if (-not $sys) {
            Fail ("파이썬 3.9 이상을 찾지 못했습니다.`n`n" +
                  "  https://www.python.org/downloads/ 에서 설치하고`n" +
                  "  설치 화면의 'Add python.exe to PATH' 를 반드시 켜주세요.")
        }
        Write-Host "  [1/3] 가상환경 생성 ($($sys -join ' '))" -ForegroundColor Green
        $head = $sys[0]
        $tail = @()
        if ($sys.Count -gt 1) { $tail = $sys[1..($sys.Count - 1)] }
        & $head @tail -m venv (Join-Path $Root '.venv')
        if (-not (Test-Path $Py)) { Fail "가상환경 생성에 실패했습니다." }
    } else {
        Write-Host "  [1/3] 가상환경 있음 — 건너뜀" -ForegroundColor DarkGray
    }

    & $Py -m pip install --quiet --upgrade pip setuptools wheel

    $idx = Get-TorchIndex
    Write-Host "  [2/3] torch 설치 — $($idx.note)" -ForegroundColor Green
    $args = @('-m', 'pip', 'install', '--index-url', $idx.url, 'torch', 'torchvision')
    if ($idx.pre) { $args = $args[0..2] + @('--pre') + $args[3..($args.Count - 1)] }
    & $Py @args
    if ($LASTEXITCODE -ne 0) {
        Fail ("torch 설치에 실패했습니다.`n`n" +
              "  requirements.txt 상단의 안내를 보고 손으로 설치한 뒤 다시 실행하세요.")
    }

    Write-Host "  [3/3] 나머지 의존성" -ForegroundColor Green
    if (Test-Path $req) {
        & $Py -m pip install -r $req
    } else {
        & $Py -m pip install ultralytics opencv-python numpy pillow PyYAML
    }
    if ($LASTEXITCODE -ne 0) { Fail "의존성 설치에 실패했습니다." }

    $chk = & $Py -c "import torch, cv2, ultralytics; print('torch %s cuda=%s' % (torch.__version__, torch.cuda.is_available()))"
    Write-Host ""
    Write-Host "  준비 완료 — $chk" -ForegroundColor Cyan
    Write-Host ""
}

Initialize-Venv
if ($SetupOnly) {
    Write-Host "  설치만 하고 종료합니다 (-SetupOnly)." -ForegroundColor DarkGray
    exit 0
}

# ---- 1. 준비물 확인 --------------------------------------------------------
if (-not (Test-Path $App)) { Fail "스튜디오 스크립트가 없습니다:`n     $App" }
$exe = if ($Console) { $Py } else { $PyW }
if (-not (Test-Path $exe)) {
    Fail ("가상환경을 찾을 수 없습니다:`n     $exe`n`n" +
          "  '스튜디오 실행 (콘솔).bat' 으로 실행하면 설치 과정을 볼 수 있습니다.")
}

# ---- 1b. 워커 모드 ---------------------------------------------------------
#  스튜디오가 워커를 자동으로 띄우므로 평소에는 쓸 일이 없다. 스튜디오 없이
#  큐만 비우고 싶을 때(밤새 돌리기 등) 쓴다.
if ($Worker) {
    $env:YOLO_AUTOINSTALL = 'false'
    Set-Location $Root
    Write-Host "  학습 워커 — 창을 열어둔 채로 두세요 (Ctrl+C 로 중단)" -ForegroundColor Cyan
    Write-Host ""
    & $Py (Join-Path $Root 'train_worker.py') @Rest
    Write-Host ""
    Read-Host "  엔터를 누르면 닫힙니다" | Out-Null
    exit 0
}

# ---- 2. 이미 떠 있는가 -----------------------------------------------------
function Get-StudioInstances {
    <# venv 의 pythonw.exe 는 실제 인터프리터를 다시 띄우는 껍데기라 한 번 실행에
       프로세스가 둘 잡힌다. 부모가 목록 안에 있는 것은 자식이므로 빼고 센다. #>
    $all = @(
        Get-CimInstance Win32_Process -Filter "Name='pythonw.exe' OR Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -like '*yolo_dataset_studio.py*' }
    )
    $ids = $all.ProcessId
    @($all | Where-Object { $_.ParentProcessId -notin $ids })
}

$running = Get-StudioInstances
if ($running.Count -gt 0) {
    $n = $running.Count
    Write-Host ""
    Write-Host "  스튜디오가 이미 $n 개 실행 중입니다 (PID $($running.ProcessId -join ', '))" -ForegroundColor Yellow
    Write-Host "  같은 작업공간을 두 창에서 열면 나중에 저장한 쪽이 앞의 편집을 덮어씁니다."
    Write-Host ""
    if (-not $Force) {
        $ans = Read-Host "  그래도 새로 띄울까요? (y = 새로 띄움 / 그 외 = 취소)"
        if ($ans -notmatch '^[yY]') {
            Write-Host "  취소했습니다." -ForegroundColor DarkGray
            exit 0
        }
    } else {
        Write-Host "  -Force 지정됨 → 그대로 진행합니다." -ForegroundColor DarkGray
    }
}

# ---- 3. 직전 실행이 오류로 끝났는가 ----------------------------------------
if (Test-Path $Log) {
    $age = (Get-Date) - (Get-Item $Log).LastWriteTime
    if ($age.TotalHours -lt 24) {
        $last = (Get-Content $Log -Tail 40 -Encoding UTF8) -join "`n"
        Write-Host ""
        Write-Host "  최근 오류 기록이 있습니다 ($([int]$age.TotalMinutes)분 전)" -ForegroundColor Yellow
        Write-Host "  $Log" -ForegroundColor DarkGray
        $head = ($last -split "`n" | Where-Object { $_ -match '^\w+Error|^\w+Exception' } | Select-Object -Last 1)
        if ($head) { Write-Host "  마지막 오류: $head" -ForegroundColor DarkGray }
        Write-Host ""
    }
}

# ---- 4. 실행 ---------------------------------------------------------------
$env:YOLO_AUTOINSTALL = 'false'   # ultralytics 가 시스템 파이썬의 torch 를 덮어쓰는 것 차단
Set-Location $Root

$argList = @($App)
if ($Rest) { $argList += $Rest }

if ($Console) {
    Write-Host "  YOLO Dataset Studio — 콘솔 모드" -ForegroundColor Cyan
    Write-Host "  $exe" -ForegroundColor DarkGray
    Write-Host ""
    & $exe @argList
    $rc = $LASTEXITCODE
    Write-Host ""
    if ($rc -ne 0) {
        Write-Host "  [!] 종료 코드 $rc" -ForegroundColor Red
        Write-Host "      기록: $Log" -ForegroundColor DarkGray
    } else {
        Write-Host "  정상 종료" -ForegroundColor DarkGray
    }
    Read-Host "  엔터를 누르면 닫힙니다" | Out-Null
} else {
    Start-Process -FilePath $exe -ArgumentList $argList -WorkingDirectory $Root
    Start-Sleep -Milliseconds 900
    $ok = Get-StudioInstances
    if ($ok.Count -eq 0) {
        Fail ("실행 직후 프로세스가 사라졌습니다.`n" +
              "  '스튜디오 실행 (콘솔).bat' 으로 다시 실행해 원인을 확인하세요.")
    }
}
