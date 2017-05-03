#!/bin/bash
#Handle spaces
OIFS="$IFS"
IFS=$'\n'
for subdir in $(find "/home/lferrer/Documents/Synthetic/First Person" -type d); do
  #subdir_relative=$(echo $subdir | sed "s:.*/home/lferrer/Documents/Synthetic/First Person/::g")
  #mkdir "~/Test/$subdir_relative"
  echo $subdir
  #for file in $(find "$subdir" -type f | head -n 10); do
   # cp "$file" "~/bar/$subdir_relative/"
  #done
done
IFS="$OIFS"