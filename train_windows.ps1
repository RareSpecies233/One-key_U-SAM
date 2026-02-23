$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Read-DefaultString {
    param(
        [string]$Prompt,
        [string]$Default
    )
    $suffix = if ([string]::IsNullOrWhiteSpace($Default)) { '' } else { " [$Default]" }
    $value = Read-Host "$Prompt$suffix"
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $Default
    }
    return $value.Trim()
}

function Read-Choice {
    param(
        [string]$Prompt,
        [string[]]$ValidValues,
        [string]$Default
    )
    while ($true) {
        $value = Read-DefaultString -Prompt $Prompt -Default $Default
        if ($ValidValues -contains $value) {
            return $value
        }
        Write-Host "输入无效，可选值: $($ValidValues -join ', ')" -ForegroundColor Yellow
    }
}

function Read-DefaultInt {
    param(
        [string]$Prompt,
        [int]$Default,
        [int]$MinValue = 1
    )
    while ($true) {
        $raw = Read-DefaultString -Prompt $Prompt -Default "$Default"
        $parsed = 0
        if ([int]::TryParse($raw, [ref]$parsed) -and $parsed -ge $MinValue) {
            return $parsed
        }
        Write-Host "请输入整数，且不小于 $MinValue。" -ForegroundColor Yellow
    }
}

function Resolve-PythonCommand {
    $projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (Test-Path $projectPython) {
        return @{ Executable = $projectPython; PrefixArgs = @() }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $pythonCmd) {
        return @{ Executable = 'python'; PrefixArgs = @() }
    }

    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $pyCmd) {
        return @{ Executable = 'py'; PrefixArgs = @('-3') }
    }

    throw '未找到可用的 Python。请先安装 Python，或在项目根目录创建 .venv。'
}

function Find-LatestCheckpoint {
    param([string]$PromptDir)

    if (-not (Test-Path $PromptDir)) {
        return $null
    }

    $checkpoints = Get-ChildItem -Path $PromptDir -Filter '*.pth' -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending

    if ($null -eq $checkpoints -or $checkpoints.Count -eq 0) {
        return $null
    }

    return $checkpoints[0].FullName
}

function Get-PromptName {
    param([int]$PromptMode)

    switch ($PromptMode) {
        0 { return 'no_prompt' }
        1 { return 'gt_boxes' }
        2 { return 'gt_pts' }
        3 { return 'gt_boxes_pts' }
        default { throw "不支持的 prompt_mode: $PromptMode" }
    }
}

Set-Location $PSScriptRoot

Write-Host '=========================================' -ForegroundColor Cyan
Write-Host ' U-SAM Windows 一键训练脚本' -ForegroundColor Cyan
Write-Host '=========================================' -ForegroundColor Cyan

try {
    $python = Resolve-PythonCommand
    $pythonExe = $python.Executable
    $pythonPrefixArgs = $python.PrefixArgs

    Write-Host "Python: $pythonExe $($pythonPrefixArgs -join ' ')" -ForegroundColor DarkGray

    $datasetChoice = Read-Choice -Prompt '选择数据集: 1=rectum, 2=word' -ValidValues @('1', '2') -Default '1'
    $dataset = if ($datasetChoice -eq '1') { 'rectum' } else { 'word' }

    $runMode = Read-Choice -Prompt '选择训练模式: 1=开始新的训练, 2=继续上次训练' -ValidValues @('1', '2') -Default '1'
    $promptMode = [int](Read-Choice -Prompt 'prompt_mode: 0=no_prompt, 1=gt_boxes, 2=gt_pts, 3=gt_boxes_pts' -ValidValues @('0', '1', '2', '3') -Default '0')
    $promptName = Get-PromptName -PromptMode $promptMode

    $defaultDataRoot = ''
    while ($true) {
        $dataRootInput = Read-DefaultString -Prompt '请输入数据集根目录 data_root（必填）' -Default $defaultDataRoot
        if ([string]::IsNullOrWhiteSpace($dataRootInput)) {
            Write-Host 'data_root 不能为空。' -ForegroundColor Yellow
            continue
        }
        $dataRoot = [System.IO.Path]::GetFullPath($dataRootInput)
        if (-not (Test-Path $dataRoot)) {
            Write-Host "目录不存在: $dataRoot" -ForegroundColor Yellow
            continue
        }
        break
    }

    $defaultOutput = if ($dataset -eq 'rectum') { '.\exp\U-SAM-Rectum' } else { '.\exp\U-SAM-Word' }
    $outputDirInput = Read-DefaultString -Prompt '输出目录 output_dir（注意是 prompt 子目录的上一级）' -Default $defaultOutput
    $outputDir = [System.IO.Path]::GetFullPath($outputDirInput)

    $epochs = Read-DefaultInt -Prompt 'epochs' -Default 100 -MinValue 1
    $batchSize = Read-DefaultInt -Prompt 'batch_size' -Default 24 -MinValue 1
    $numWorkers = Read-DefaultInt -Prompt 'num_workers' -Default 2 -MinValue 0
    $device = Read-DefaultString -Prompt 'device (cuda/cpu)' -Default 'cuda'

    $resumePath = ''
    $promptDir = Join-Path $outputDir "prompt=$promptName"

    if ($runMode -eq '2') {
        $resumeSelect = Read-Choice -Prompt '续训 checkpoint: 1=自动查找最新, 2=手动输入路径' -ValidValues @('1', '2') -Default '1'

        if ($resumeSelect -eq '1') {
            $latestCkpt = Find-LatestCheckpoint -PromptDir $promptDir
            if ($null -eq $latestCkpt) {
                Write-Host "未在 $promptDir 找到 checkpoint，将切换为手动输入。" -ForegroundColor Yellow
                $resumeSelect = '2'
            }
            else {
                $resumePath = $latestCkpt
                Write-Host "已自动选择: $resumePath" -ForegroundColor Green
            }
        }

        if ($resumeSelect -eq '2') {
            while ($true) {
                $manualResume = Read-DefaultString -Prompt '请输入 checkpoint 文件路径(.pth)' -Default ''
                if ([string]::IsNullOrWhiteSpace($manualResume)) {
                    Write-Host 'checkpoint 路径不能为空。' -ForegroundColor Yellow
                    continue
                }

                $candidate = [System.IO.Path]::GetFullPath($manualResume)
                if (-not (Test-Path $candidate)) {
                    Write-Host "文件不存在: $candidate" -ForegroundColor Yellow
                    continue
                }

                $resumePath = $candidate
                break
            }
        }
    }

    if (-not (Test-Path $outputDir)) {
        New-Item -Path $outputDir -ItemType Directory -Force | Out-Null
    }

    $logsDir = Join-Path $outputDir 'launcher_logs'
    if (-not (Test-Path $logsDir)) {
        New-Item -Path $logsDir -ItemType Directory -Force | Out-Null
    }

    $sessionId = Get-Date -Format 'yyyyMMdd_HHmmss'
    $sessionLog = Join-Path $logsDir ("train_$sessionId.log")
    $sessionMeta = Join-Path $logsDir ("train_$sessionId.json")

    $trainArgs = @(
        'u-sam.py'
        '--epochs', "$epochs"
        '--batch_size', "$batchSize"
        '--dataset', $dataset
        '--data_root', $dataRoot
        '--output_dir', $outputDir
        '--prompt_mode', "$promptMode"
        '--num_workers', "$numWorkers"
        '--device', $device
    )

    if (-not [string]::IsNullOrWhiteSpace($resumePath)) {
        $trainArgs += @('--resume', $resumePath)
    }

    Write-Host ''
    Write-Host '即将执行训练命令：' -ForegroundColor Cyan
    Write-Host "$pythonExe $($pythonPrefixArgs -join ' ') $($trainArgs -join ' ')" -ForegroundColor DarkGray
    Write-Host "日志文件: $sessionLog" -ForegroundColor DarkGray
    Write-Host ''

    $confirmStart = Read-Choice -Prompt '确认开始训练？1=开始, 2=取消' -ValidValues @('1', '2') -Default '1'
    if ($confirmStart -eq '2') {
        Write-Host '已取消训练。' -ForegroundColor Yellow
        exit 0
    }

    $startTime = Get-Date
    & $pythonExe @pythonPrefixArgs @trainArgs 2>&1 | Tee-Object -FilePath $sessionLog
    $exitCode = $LASTEXITCODE
    $endTime = Get-Date
    $duration = New-TimeSpan -Start $startTime -End $endTime

    $latestMetrics = $null
    $trainLogPath = Join-Path $promptDir 'log.txt'
    if (Test-Path $trainLogPath) {
        try {
            $tailLine = Get-Content -Path $trainLogPath -Tail 1 -ErrorAction Stop
            if (-not [string]::IsNullOrWhiteSpace($tailLine)) {
                $latestMetrics = $tailLine | ConvertFrom-Json
            }
        }
        catch {
            $latestMetrics = $null
        }
    }

    $checkpointCount = 0
    if (Test-Path $promptDir) {
        $checkpointCount = @(Get-ChildItem -Path $promptDir -Filter '*.pth' -File -ErrorAction SilentlyContinue).Count
    }

    $summary = [ordered]@{
        session_id = $sessionId
        exit_code = $exitCode
        dataset = $dataset
        run_mode = if ($runMode -eq '1') { 'new' } else { 'resume' }
        data_root = $dataRoot
        output_dir = $outputDir
        prompt_dir = $promptDir
        prompt_mode = $promptMode
        prompt_name = $promptName
        resume = $resumePath
        epochs = $epochs
        batch_size = $batchSize
        num_workers = $numWorkers
        device = $device
        start_time = $startTime.ToString('yyyy-MM-dd HH:mm:ss')
        end_time = $endTime.ToString('yyyy-MM-dd HH:mm:ss')
        duration = $duration.ToString()
        launcher_log = $sessionLog
        train_log = $trainLogPath
        checkpoint_count = $checkpointCount
        mean_dice = if ($null -ne $latestMetrics -and $latestMetrics.PSObject.Properties.Name -contains 'test_mean_dice') { [double]$latestMetrics.test_mean_dice } else { $null }
        miou = if ($null -ne $latestMetrics -and $latestMetrics.PSObject.Properties.Name -contains 'test_miou') { [double]$latestMetrics.test_miou } else { $null }
        last_epoch = if ($null -ne $latestMetrics -and $latestMetrics.PSObject.Properties.Name -contains 'epoch') { [int]$latestMetrics.epoch } else { $null }
    }

    $summary | ConvertTo-Json -Depth 6 | Set-Content -Path $sessionMeta -Encoding UTF8

    Write-Host ''
    Write-Host '=========================================' -ForegroundColor Cyan
    Write-Host ' 本次训练信息' -ForegroundColor Cyan
    Write-Host '=========================================' -ForegroundColor Cyan
    Write-Host ("退出码: {0}" -f $summary.exit_code)
    Write-Host ("训练模式: {0}" -f $summary.run_mode)
    Write-Host ("数据集: {0}" -f $summary.dataset)
    Write-Host ("prompt: {0}" -f $summary.prompt_name)
    Write-Host ("持续时间: {0}" -f $summary.duration)
    Write-Host ("输出目录: {0}" -f $summary.prompt_dir)
    Write-Host ("checkpoint 数量: {0}" -f $summary.checkpoint_count)
    if ($null -ne $summary.last_epoch) {
        Write-Host ("最后 epoch: {0}" -f $summary.last_epoch)
    }
    if ($null -ne $summary.mean_dice) {
        Write-Host ("最后 mean_dice: {0:N6}" -f $summary.mean_dice)
    }
    if ($null -ne $summary.miou) {
        Write-Host ("最后 miou: {0:N6}" -f $summary.miou)
    }
    Write-Host ("训练日志: {0}" -f $summary.launcher_log)
    Write-Host ("会话摘要: {0}" -f $sessionMeta)

    while ($true) {
        $confirmExit = Read-Choice -Prompt '是否确认结束？1=确认退出, 2=查看最后20行日志后再决定' -ValidValues @('1', '2') -Default '1'
        if ($confirmExit -eq '1') {
            break
        }
        if (Test-Path $sessionLog) {
            Write-Host ''
            Write-Host '------ 日志尾部(20行) ------' -ForegroundColor DarkGray
            Get-Content -Path $sessionLog -Tail 20
            Write-Host '-----------------------------' -ForegroundColor DarkGray
            Write-Host ''
        }
    }

    if ($summary.exit_code -ne 0) {
        exit $summary.exit_code
    }
}
catch {
    Write-Host "脚本执行失败: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
