# AEROTHON 26-27
AEOLUS
```
scp szyfrowac@10.42.0.247:~/aerothon_here_we_come.tar.gz ~/
find /media/$USER ~ -maxdepth 3 -name "aerothon_here_we_come.tar.gz" 2>/dev/null
rm -rf ~/aerothon_here_we_come ~/aerothon_radio_gcs
tar xzf /PATH/FROM/STEP2/aerothon_here_we_come.tar.gz -C ~
~/aerothon_here_we_come/install.sh
sudo usermod -aG dialout $USER



#switcher app
 ~/aerothon_here_we_come/aerothon switcher
 
 #open at http://127.0.0.1:8079
 
 #command to run
 aerothon switcher
 
 
 

```


 # ON JETSON

-  aerothon-router connects PIXHAWK over USB to everything else
- aerothon-mav-agent : radio agent takes launch + start + geofence from lenovo,  sends status, VIO and boards back
- gcs-dashboard (8080)
- mission container : isaac_ros_dev-aarch64-pyzbar-container
- if something on jetson not launched yet (aerothon-router shows activating until pixhawk is plugged in)
```
systemctl --user is-active aerothon-router aerothon-mav-agent gcs-dashboard     # all should say active
systemctl --user restart aerothon-router aerothon-mav-agent gcs-dashboard       # if not
```


ON LENOVO
- mission stack : started by pressing launch on lenovo, over radio
- 8084 dashboard : only for lab testing over cable 
- aerothon VIO container : started by launch
```
~/workspaces/isaac_ros-dev/src/launch_all/scripts/gcs_aruco.sh
start
```


TO START MISSION A (onboard + AruCO)
```
aerothon switcher
#OR if doesn't launch
 ~/aerothon_here_we_come/aerothon switcher
```

To start an app directly without the switcher :
- Final mission (radio) : 
	- aerothon onboard Aruco 
	- aerothon onboard qr
- Offboard on the laptop
	- aerothon offboard aruco drop /dev/ttyUSB0:57600 /dev/video2
	-  aerothon offboard qr drop /dev/ttyUSB0:57600 /dev/video2
	-  aerothon offboard aruco nodrop /dev/ttyUSB0:57600 /dev/video2
	-  aerothon offboard qr nodrop /dev/ttyUSB0:57600 /dev/video2
- sim testing 
	- aerothon sim aruco
	- aerothon sim qr
- stop whatever is running
	- aerothon stop
- what is running + log
	- aerothon status
	- aerothon log



