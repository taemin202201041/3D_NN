# Export every slide of a .pptx to PNG with PowerPoint (for visual QA).  usage: powershell -File export_slides.ps1 <pptx> <outdir>
param([string]$Pptx, [string]$Out)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $Out | Out-Null
$app = New-Object -ComObject PowerPoint.Application
try {
    $pres = $app.Presentations.Open((Resolve-Path $Pptx).Path, $true, $false, $false)
    $pres.Export((Resolve-Path $Out).Path, "PNG", 1600, 900)
    $pres.Close()
} finally {
    $app.Quit()
}
Get-ChildItem $Out -Filter *.PNG | Sort-Object Name | Select-Object -ExpandProperty FullName
