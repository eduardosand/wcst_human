#!/bin/bash

# Remote server details
remote_user="esandoval"
remote_host="dev0.uwcnc.net"

# Default directory
starting_path="/mnt/U19_NAS/human_data/wcst"

# template path
template_path="subject/sess-session/raw"

local_path="/home/eduardo/wcst_human/data"
# Looping through each subject
for subject in IR85 IR87 IR94 IR95 DA9
#for subject in IR84 IR98 IR99 IR86 IR100
do
  echo $subject
  # replace all occurrences of subject with current
#  modified_subject_path="${template_path//subject/participant}"
  modified_subject_path=$(echo "$template_path" | sed "s/subject/$subject/")
  echo $modified_subject_path
  for session in 1 2 3 4
  do
    modified_session_path=$(echo "$modified_subject_path" | sed "s/session/$session/")
#    modified_session_path="${modified_subject_path//session/$session}"
    full_path="$starting_path/$modified_session_path"
    echo "$full_path"
    modified_local_path=$(echo "$modified_session_path" | sed "s/raw//")
    full_local_path="$local_path/$modified_local_path"
    # check if full_path exists
#    if [ -e "$full_path" ]; then
#      echo nice
    mkdir -p $full_local_path
    scp -r -v "$remote_user@$remote_host:$full_path" "$full_local_path"
#    fi
  done
done