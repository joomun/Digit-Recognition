@echo off
SET /A count=0
SET /A max_runs=100

:loop
if %count% lss %max_runs% (
    java -cp bin Main
    SET /A count+=1
    goto loop
)
echo Finished running %max_runs% times.
