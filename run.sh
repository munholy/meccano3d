#!/bin/bash

for ((i=1; i<=10000; i++))
do
    blenderproc run main.py parts
done