@echo off
set OPENVIKING_CONFIG_FILE=D:\Code\cursorProject\OpenViking\benchmark\memrouter_embedded\config\ov+graph.conf
"D:\ProgramFiles\anaconda3\envs\openviking\python.exe" -m openviking.server.bootstrap --config "%OPENVIKING_CONFIG_FILE%"
