@echo off
title 3x4 va 10x15 Hujjat Boti
chcp 65001 > nul
echo ==============================================
echo    3x4 va 10x15 Fotosurat Telegram Boti
echo ==============================================
echo.
echo Virtual muhit faollashtirilmoqda...
call venv\Scripts\activate
echo.
echo Bot ishga tushirilmoqda...
echo Botingiz: https://t.me/dokumentpdfbot
echo To'xtatish uchun: Ctrl + C
echo.
python bot.py
pause
