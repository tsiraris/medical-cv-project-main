param(
  [int]$Seconds = 60,
  [string]$Url = "http://localhost:8000/predict"
)
$start = Get-Date
while (((Get-Date) - $start).TotalSeconds -lt $Seconds) {
  $body = @{ items = @(
    @{ f1=5.1; f2=3.5; f3=1.4; f4=0.2 },
    @{ f1=6.2; f2=3.0; f3=4.5; f4=1.5 }
  )} | ConvertTo-Json
  try { Invoke-RestMethod -Method Post -Uri $Url -Body $body -ContentType 'application/json' | Out-Null } catch {}
}
