#!/bin/bash
# Remote server details
remote_user="esandoval"
remote_host="dev0.uwcnc.net"

# Default directory
starting_path="/mnt/U19_NAS/human_data/wcst"

# template path
template_path="subject/sess-session/behavior"

local_path="/home/eduardo/WCST_Human"
# Looping through each subject
for subject in IR82 IR84 IR85 IR86 IR87 IR94 IR95 IR98 IR99 IR100 IR101 DA9
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
    remote_folder="$starting_path/$modified_session_path"
    ssh "$remote_user@$remote_host" "[ -d \"$remote_folder\" ]" 2>/dev/null
    # outer if checks if folder exists server side
    if [ $? -eq 0 ]; then
      echo "Folder exists: $remote_folder"
      echo "$full_path"
      modified_local_path=$(echo "$modified_session_path" | sed "s/behavior//")
      full_local_path="$local_path/$modified_local_path"
      # check if full_path exists
  #    if [ -e "$full_path" ]; then
  #      echo nice
      mkdir -p $full_local_path
      scp -r -v "$remote_user@$remote_host:$remote_folder" "$full_local_path"
  #    fi
    fi
  done
done