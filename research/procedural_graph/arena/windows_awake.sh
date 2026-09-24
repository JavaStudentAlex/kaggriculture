#!/usr/bin/env bash
# Keep the Windows host of this WSL machine from idle-sleeping while Colab VMs run, so the
# CLI's keep-alive daemons keep pinging (AGENTS.md section 3.1). It holds an
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED request from PowerShell in tmux session kagg-awake,
# for at most HOURS hours (default 4). The display may still turn off. Closing a laptop's
# lid or choosing Sleep still sleeps the machine.
#
#   bash arena/windows_awake.sh [HOURS] [LOG]     # start (log: KAGG_WAKELOCK_ON pid=...)
#   bash arena/windows_awake.sh stop              # release it early
set -euo pipefail
if [[ "${1:-}" == stop ]]; then
    powershell.exe -NoProfile -Command 'New-Item -ItemType File -Force "$env:TEMP\kagg_wakelock_stop" | Out-Null'
    echo "stop requested; the lock is released within 30 s"
    exit 0
fi
hours=${1:-4}
log=${2:-/tmp/kagg-awake.log}
tmux has-session -t kagg-awake 2>/dev/null && { echo "kagg-awake is already running"; exit 0; }
enc=$(HOURS="$hours" python3 - <<'EOF'
import base64, os
ps = r'''
$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$k = Add-Type -MemberDefinition $sig -Name 'KaggAwake' -Namespace 'Win32' -PassThru
$null = $k::SetThreadExecutionState([uint32]2147483649)
"KAGG_WAKELOCK_ON pid=$PID $(Get-Date -Format HH:mm:ss)"
$end = (Get-Date).AddHours(HOURS)
while ((Get-Date) -lt $end) {
  if (Test-Path "$env:TEMP\kagg_wakelock_stop") { Remove-Item "$env:TEMP\kagg_wakelock_stop"; break }
  Start-Sleep -Seconds 30
}
$null = $k::SetThreadExecutionState([uint32]2147483648)
"KAGG_WAKELOCK_OFF $(Get-Date -Format HH:mm:ss)"
'''.replace('HOURS', str(float(os.environ['HOURS'])))
print(base64.b64encode(ps.encode('utf-16-le')).decode())
EOF
)
tmux new -d -s kagg-awake "powershell.exe -NoProfile -EncodedCommand $enc > $log 2>&1; echo AWAKE_EXIT=\$? >> $log"
echo "kagg-awake started for up to $hours h; log $log"
