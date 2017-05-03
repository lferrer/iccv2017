for d in /home/lferrer/Downloads/Full\ Res\ Videos/First\ Person/*;
    do
        ffmpeg -i "$d/FirstPerson-None.mp4" "$d/%d.jpg"
    done;
