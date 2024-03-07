import requests

def download_file(url, filename):
    # Startet die Anfrage und holt die Daten
    response = requests.get(url, allow_redirects=True)
    # Speichert die heruntergeladenen Daten in einer Datei
    open(filename, 'wb').write(response.content)
    print(f"{filename} wurde erfolgreich heruntergeladen.")

# URLs für die YOLOv3-Gewichts- und Konfigurationsdateien
weights_url = "https://pjreddie.com/media/files/yolov3.weights"
cfg_url = "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true"
coco_names_url = "https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true"

# Speicherpfade für die heruntergeladenen Dateien
weights_path = "yolov3.weights"
cfg_path = "yolov3.cfg"
coco_names_path = "coco.names"

# Download der Dateien
download_file(weights_url, weights_path)
download_file(cfg_url, cfg_path)
download_file(coco_names_url, coco_names_path)
