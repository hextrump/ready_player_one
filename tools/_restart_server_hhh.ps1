# 重启测谎服务端 — Win32_Process.Create + cmd /c 重定向 (全分离, 跨 ssh 存活)
$ErrorActionPreference = 'Stop'

$py    = 'C:\Users\heyas\.cache\rimagination-notes\qwen3-asr-venv\Scripts\python.exe'
$srv   = 'C:\Users\heyas\Documents\code\ready_player_one\tools\lie_detect_server.py'
$repo  = 'C:\Users\heyas\Documents\code\lie-detector'
$work  = 'C:\Users\heyas\Documents\code\ready_player_one'
$out   = 'C:\Users\heyas\tmp_diag\server_out.log'
$err   = 'C:\Users\heyas\tmp_diag\server_err.log'

# 0) 关掉现有 lie_detect_server 进程
foreach ($p in Get-CimInstance Win32_Process) {
    if ($p.CommandLine -match 'lie_detect_server') {
        Write-Output ("kill PID {0}" -f $p.ProcessId)
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Seconds 1

# 1) 构造 cmd 命令行 (cmd.exe 处理 > 重定向, Create 全分离)
$inner   = '"' + $py + '" "' + $srv + '" --repo "' + $repo + '" > "' + $out + '" 2> "' + $err + '"'
$cmdline = 'cmd.exe /c "' + $inner + '"'
Write-Output ("cmdline: {0}" -f $cmdline)

$res = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $cmdline
    CurrentDirectory = $work
}
Write-Output ("Create ReturnValue={0} PID={1}" -f $res.ReturnValue, $res.ProcessId)

# 2) 等预热 + 端口就绪 (模型加载 ~14s)
for ($i = 1; $i -le 20; $i++) {
    Start-Sleep -Seconds 3
    $listening = Get-NetTCPConnection -LocalPort 8600 -State Listen -ErrorAction SilentlyContinue
    if ($listening) {
        Write-Output ("LISTENING at t={0}s" -f ($i * 3))
        break
    }
}

# 3) /health 自检
try {
    $h = Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 'http://127.0.0.1:8600/health'
    Write-Output ("health: {0}" -f $h.Content)
} catch {
    Write-Output ("health fail: {0}" -f $_.Exception.Message)
}

Write-Output '--- err tail ---'
Get-Content $err -Tail 8 -ErrorAction SilentlyContinue
