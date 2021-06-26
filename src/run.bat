python word2vec.py
python tokenization/tokenization.py
python parsing/run.py
python language_model.py 
python fine_tuning.py Bert
python fine_tuning.py GPT2

@REM @echo off
@REM rem    Run this file on the command line of an environment that contains "python" in path
@REM rem    For example, in the terminal of your IDE
@REM rem    Or in the correct environment of your anaconda prompt

@REM if "%1%"=="word2vec" (
@REM     python word2vec.py
@REM ) else if "%1%"=="tokenization" (
@REM     python tokenization/tokenization.py
@REM ) else if "%1%"=="parsing" (
@REM     python parsing/run.py 
@REM ) else if "%1%"=="language_model" (
@REM     python language_model.py 
@REM ) else if "%1%"=="fine_tuning-bert" (
@REM     python fine_tuning.py Bert
@REM ) else if "%1%"=="fine_tuning-gpt2"(
@REM     python fine_tuning.py GPT2
@REM ) else (
@REM     echo Invalid Option Selected
@REM )
