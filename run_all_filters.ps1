# PowerShell script to run all filter files one by one with GPU support
Write-Host "Running all filter files with GPU support..." -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green

# Set CUDA environment variables explicitly
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"

# Get all filter files
$filterDir = "data\filters_to_prune"
$filterFiles = Get-ChildItem -Path $filterDir -Filter "*.txt"

# Configuration
$epochs = 10
$batchSize = 64
$learningRate = 0.001
$logDir = "logs"

# Display GPU information
Write-Host "GPU Configuration:" -ForegroundColor Cyan
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,noheader
    Write-Host $gpuInfo -ForegroundColor Cyan
} catch {
    Write-Host "Could not get GPU information. Make sure nvidia-smi is available." -ForegroundColor Yellow
}
Write-Host "======================================================" -ForegroundColor Green

# Create logs directory if it doesn't exist
if (-not (Test-Path -Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
    Write-Host "Created logs directory: $logDir"
}

# Display the number of filter files found
Write-Host "Found $($filterFiles.Count) filter files to process."

# Initialize counters
$totalFiles = $filterFiles.Count
$completedFiles = 0
$successfulFiles = 0
$failedFiles = 0

# Process each filter file
foreach ($file in $filterFiles) {
    $filterPath = $file.FullName
    $filterName = $file.BaseName
    $logFile = Join-Path -Path $logDir -ChildPath "$filterName.log"

    $completedFiles++

    Write-Host ""
    Write-Host "======================================================" -ForegroundColor Cyan
    Write-Host "Processing filter $completedFiles of $totalFiles`: $filterName" -ForegroundColor Cyan
    Write-Host "Filter path: $filterPath"
    Write-Host "Log will be saved to: $logFile"

    # Display GPU memory before running
    try {
        $gpuMemBefore = nvidia-smi --query-gpu=memory.used --format=csv,noheader
        Write-Host "GPU memory before: $gpuMemBefore" -ForegroundColor Magenta
    } catch {
        Write-Host "Could not get GPU memory information." -ForegroundColor Yellow
    }

    # Run the Python process directly
    Write-Host "Starting finetuning process..." -ForegroundColor Yellow

    # Run the command and redirect output to log file
    $startTime = Get-Date
    Write-Host "Start time: $startTime" -ForegroundColor Cyan

    # Execute the command and redirect output to log file
    # Add --no-show-plot flag to prevent interactive plots from halting the batch process
    try {
        python pruner/finetune_simplified.py --filters "$filterPath" --epochs $epochs --batch-size $batchSize --lr $learningRate --no-show-plot *> $logFile

        # Check if the process completed successfully
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Process completed successfully" -ForegroundColor Green
            $successfulFiles++
        } else {
            Write-Host "Process failed with exit code: $LASTEXITCODE" -ForegroundColor Red
            Write-Host "Check log file for details: $logFile"
            $failedFiles++
        }
    } catch {
        Write-Host "Error running Python script: $_" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        $failedFiles++
    }

    $endTime = Get-Date
    $duration = $endTime - $startTime
    Write-Host "End time: $endTime" -ForegroundColor Cyan
    Write-Host "Duration: $($duration.ToString())" -ForegroundColor Cyan

    # Display GPU memory after running
    try {
        $gpuMemAfter = nvidia-smi --query-gpu=memory.used --format=csv,noheader
        Write-Host "GPU memory after: $gpuMemAfter" -ForegroundColor Magenta
    } catch {
        Write-Host "Could not get GPU memory information." -ForegroundColor Yellow
    }

    # Clear GPU memory between runs
    Write-Host "Waiting 10 seconds to ensure GPU memory is released..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10

    # Try to force garbage collection
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()

    Write-Host "Completed processing filter: $filterName" -ForegroundColor Cyan
    Write-Host "Progress: $completedFiles/$totalFiles ($([math]::Round(($completedFiles/$totalFiles)*100))%)" -ForegroundColor Green
    Write-Host "======================================================" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host "All filter files processed" -ForegroundColor Green
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  Total files processed: $totalFiles" -ForegroundColor Cyan
Write-Host "  Successful: $successfulFiles" -ForegroundColor Green
Write-Host "  Failed: $failedFiles" -ForegroundColor $(if ($failedFiles -gt 0) { "Red" } else { "Green" })
Write-Host "  Success rate: $([math]::Round(($successfulFiles/$totalFiles)*100))%" -ForegroundColor $(if ($successfulFiles -eq $totalFiles) { "Green" } else { "Yellow" })
Write-Host "Logs are available in the $logDir directory" -ForegroundColor Cyan
Write-Host "Checkpoints are available in the checkpoints directory" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Green

# List the log files
Write-Host "Log files:" -ForegroundColor Cyan
Get-ChildItem -Path $logDir -Filter "*.log" | ForEach-Object {
    Write-Host "  $($_.Name)" -ForegroundColor $(if ($_.Length -gt 0) { "Green" } else { "Red" })
}

Read-Host "Press Enter to continue..."
