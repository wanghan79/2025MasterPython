@echo off
echo 蛋白质-配体相互作用分析
echo ===================================

set PDB_FILE=1ida.pdb
set OUTPUT_DIR=1ida_output

if not "%1"=="" (
    set PDB_FILE=%1
)

if not "%2"=="" (
    set OUTPUT_DIR=%2
)

set RADIUS=10.0

if not "%3"=="" (
    set RADIUS=%3
)

echo PDB文件: %PDB_FILE%
echo 输出目录: %OUTPUT_DIR%
echo 结合位点半径: %RADIUS%

python run_analysis.py --pdb "%PDB_FILE%" --output "%OUTPUT_DIR%" --radius %RADIUS%

if %ERRORLEVEL% NEQ 0 (
    echo 错误：分析失败
    exit /b 1
)

echo 分析完成！
echo 结果保存到 %OUTPUT_DIR%