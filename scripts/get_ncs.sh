#!/bin/bash

# Remote server details
remote_user="esandoval"
remote_host="dev0.uwcnc.net"

# Default directory we start from in server
starting_path="/mnt/U19_NAS/human_data/wcst"
#remote_folder="/mnt/U19_NAS/human_data/wcst/IR87"
# template path, using directory structure with helping look through subjects and sessions
# keep in mind this might change if we look for LFP, or behavior because the last part of directory is specific
template_path="subject/sess-session/raw"

# type of files we're looking for
microwires_files="m*.ncs"
photodiode_files="photo*.ncs"
event_file="Events.nev"

# Local file structure
local_path="/home/eduardo/WCST_Human"

# Looping through each subject
#for subject in IR82 IR84 IR85 IR86 IR87 IR94 IR95 IR98 IR99 IR100 IR101 DA9
for subject in IR87
do
  echo $subject
  # replace all occurrences of subject with current
#  modified_subject_path="${template_path//subject/participant}"
  modified_subject_path=$(echo "$template_path" | sed "s/subject/$subject/")
  echo $modified_subject_path
  for session in 1 2 3 4
  do
    modified_session_path=$(echo $modified_subject_path | sed "s/session/$session/")
    remote_folder="$starting_path/$modified_session_path" # this is like the remote_folder
    echo "$remote_folder"
    modified_local_path=$(echo $modified_session_path | sed "s/behavior//")
    full_local_folder="$local_path/$modified_local_path"
    # check if full_path exists
    # Check if folder exists on remote server
    # the second part of this command handles the error and generates a non-zero exit code
    ssh "$remote_user@$remote_host" "[ -d \"$remote_folder\" ]" 2>/dev/null
    # outer if checks if folder exists server side
    if [ $? -eq 0 ]; then
      echo "Folder exists: $remote_folder"
      # inner if checks if folder exists locally
      if [ -d "$full_local_folder" ]; then
        echo "Folder exists: $full_local_folder"
      else
        echo "Folder does not exist: $full_local_folder"
        mkdir -p $full_local_folder
      fi
      ssh "$remote_user@$remote_host" "find \"$remote_folder\" -maxdepth 1 -type f -name \"$microwires_files\"" 2>/dev/null | while IFS= read -r remote_file; do
        echo "Match 1 found: \"$remote_file\" "
        echo "$full_local_folder"
        scp "$remote_user@$remote_host:$remote_file" "$full_local_folder"
      done
      ssh "$remote_user@$remote_host" "find \"$remote_folder\" -maxdepth 1 -type f -name \"$photodiode_files\"" 2>/dev/null | while IFS= read -r remote_file; do
        echo "Match 2 found: \"$remote_file\" "
        scp "$remote_user@$remote_host:$remote_file" "$full_local_folder"
      done
      ssh "$remote_user@$remote_host" "find \"$remote_folder\" -maxdepth 1 -type f -name \"$event_file\"" 2>/dev/null | while IFS= read -r remote_file; do
        echo "Match 2 found: \"$remote_file\" "
        scp "$remote_user@$remote_host:$remote_file" "$full_local_folder"
      done
    fi
  done
done