@echo off
cd /d %~dp0
call run_gtm.bat
call run_gtexport.bat
exit