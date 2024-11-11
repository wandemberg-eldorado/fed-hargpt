#!/bin/bash

exp=$1
search_dir=$2
for entry in "$search_dir"/*
do
  echo "$entry"
  name=${entry##*/}
  python3 client_har_fed.py -u "$name" -e "$exp" &
done